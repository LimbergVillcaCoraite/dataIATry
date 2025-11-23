"""Stacking OOF: LGBM + CatBoost base models, meta LGBM.

Proceso:
- carga datos y aplica pipeline (features + lags)
- split temporal train/test (80/20)
- sobre X_train genera OOF preds para LGBM y CatBoost usando TimeSeriesSplit
- entrena modelos base completos sobre X_train y predice X_test
- entrena meta-LGBM sobre OOF preds y evalúa en X_test
- guarda `models/model_stacked.joblib` y métricas si mejora la referencia `model_lgbm.metrics.json`
"""
from pathlib import Path
import json
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from src.preprocessing import build_features, create_lag_features, prepare_X_y


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset_alquiler.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)


def time_split_df(df: pd.DataFrame, date_col: str = 'fecha', frac: float = 0.8):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col).reset_index(drop=True)
        split = int(len(df) * frac)
        return df.iloc[:split], df.iloc[split:]
    else:
        train = df.sample(frac=frac, random_state=42)
        return train, df.drop(train.index)


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {'rmse': rmse, 'mae': mae}


def run():
    print('Cargando datos...')
    df = pd.read_csv(DATA)

    print('Aplicando features y lags...')
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    print('Split temporal...')
    train_df, test_df = time_split_df(df)

    X_train, y_train = prepare_X_y(train_df, exclude=['u_casuales', 'u_registrados'])
    X_test, y_test = prepare_X_y(test_df, exclude=['u_casuales', 'u_registrados'])

    tscv = TimeSeriesSplit(n_splits=5)

    # OOF arrays
    oof_lgb = np.zeros(len(X_train))
    oof_cat = np.zeros(len(X_train))

    fold = 0
    for tr_idx, val_idx in tscv.split(X_train):
        fold += 1
        print(f'Fold {fold}')
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # LGBM base
        lgbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
        lgbm.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgbm.predict(X_val)

        # CatBoost base
        if CatBoostRegressor is not None:
            cat = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0)
            # detect categorical cols in original train_df
            cat_cols = [c for c in ['temporada', 'feriado', 'dia_semana', 'clima'] if c in train_df.columns]
            # map categorical names to indices in X_tr (if present)
            cat_feat_idx = [i for i, col in enumerate(X_tr.columns) if col in cat_cols]
            try:
                cat.fit(X_tr, y_tr, cat_features=cat_feat_idx)
                oof_cat[val_idx] = cat.predict(X_val)
            except Exception:
                # fallback: train without cat_features
                cat.fit(X_tr, y_tr)
                oof_cat[val_idx] = cat.predict(X_val)
        else:
            oof_cat[val_idx] = np.mean(y_tr)

    # Train base models on full X_train to predict X_test
    print('Entrenando base models completos...')
    lgb_full = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    lgb_full.fit(X_train, y_train)
    pred_lgb_test = lgb_full.predict(X_test)

    if CatBoostRegressor is not None:
        cat_full = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0)
        cat_feat_idx = [i for i, col in enumerate(X_train.columns) if col in ['temporada', 'feriado', 'dia_semana', 'clima']]
        try:
            cat_full.fit(X_train, y_train, cat_features=cat_feat_idx)
            pred_cat_test = cat_full.predict(X_test)
        except Exception:
            cat_full.fit(X_train, y_train)
            pred_cat_test = cat_full.predict(X_test)
    else:
        pred_cat_test = np.repeat(y_train.median(), len(X_test))

    # Meta features for training and test
    X_meta_train = pd.DataFrame({'lgb_oof': oof_lgb, 'cat_oof': oof_cat})
    X_meta_test = pd.DataFrame({'lgb_pred': pred_lgb_test, 'cat_pred': pred_cat_test})

    # Train meta model
    meta = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    meta.fit(X_meta_train, y_train)
    pred_meta = meta.predict(X_meta_test)

    metrics = evaluate(y_test, pred_meta)
    print('Stacked model metrics:', metrics)

    # Baseline reference: try to read model_lgbm.metrics.json
    ref_path = MODELS / 'model_lgbm.metrics.json'
    ref_rmse = None
    if ref_path.exists():
        try:
            with open(ref_path, 'r') as f:
                ref = json.load(f)
                ref_rmse = float(ref.get('rmse'))
        except Exception:
            ref_rmse = None

    print('Reference LGBM RMSE:', ref_rmse)

    if ref_rmse is None or metrics['rmse'] < ref_rmse:
        out_model = MODELS / 'model_stacked.joblib'
        joblib.dump(meta, out_model)
        with open(out_model.with_suffix('.metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print('Stacked model saved to', out_model)
    else:
        print('Stacked model did not improve over reference; not saved.')


if __name__ == '__main__':
    run()
