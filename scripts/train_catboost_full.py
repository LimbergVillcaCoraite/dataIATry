"""Entrena CatBoost en todo el dataset y guarda el artefacto final.

1. Carga `dataset_alquiler.csv`.
2. Aplica `build_features` y `create_lag_features`.
3. Evalúa con TimeSeriesSplit (CV) para reportar RMSE/MAE.
4. Reentrena en todo el conjunto y guarda `models/model_catboost_final.joblib` y métricas.
"""
from pathlib import Path
import json
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor

from src.preprocessing import build_features, create_lag_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_alquiler.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def prepare_data(df, target='total_alquileres', exclude=None):
    if exclude is None:
        exclude = []
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    if target not in df.columns:
        raise RuntimeError('Target not found')

    y = df[target]
    X_raw = df.drop(columns=[c for c in [target] + exclude if c in df.columns])

    mask = ~y.isna()
    X_raw = X_raw.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # detect categorical columns
    cat_cols = list(X_raw.select_dtypes(include=['category']).columns)

    # Drop raw 'fecha' column if present — no necesitamos pasar timestamp raw to CatBoost
    if 'fecha' in X_raw.columns:
        X_raw = X_raw.drop(columns=['fecha'])

    # numeric imputation
    X_num = X_raw.select_dtypes(include=[np.number]).copy()
    X_raw[X_num.columns] = X_num.fillna(X_num.median())

    # fill categorical missing
    for c in cat_cols:
        try:
            X_raw[c] = X_raw[c].astype(object).fillna('__MISSING__')
        except Exception:
            X_raw[c] = X_raw[c].fillna('__MISSING__')

    X_final = X_raw.copy()

    return X_final, y, cat_cols


def main():
    print('Cargando datos...')
    df = pd.read_csv(DATA)

    print('Preparando data...')
    X, y, cat_cols = prepare_data(df, exclude=['u_casuales', 'u_registrados'])

    print('Cat columns:', cat_cols)

    # Use best params from previous HPO if available
    hpo_metrics = MODELS / 'model_catboost_hpo.metrics.json'
    params = dict(iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3)
    if hpo_metrics.exists():
        try:
            with open(hpo_metrics, 'r') as f:
                j = json.load(f)
                bp = j.get('best_params', {})
                params.update(bp)
        except Exception:
            pass

    print('Training CatBoost with params:', params)

    model = CatBoostRegressor(loss_function='RMSE', verbose=50, random_seed=42, **params)

    # Estimate CV performance with manual TimeSeriesSplit (so we can pass cat_features)
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    maes = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        est = CatBoostRegressor(loss_function='RMSE', verbose=0, random_seed=42, **params)
        est.fit(X_tr, y_tr, cat_features=cat_cols)
        pred = est.predict(X_te)
        rmses.append(sqrt(mean_squared_error(y_te, pred)))
        maes.append(np.mean(np.abs(y_te - pred)))

    mean_rmse = float(np.mean(rmses))
    mean_mae = float(np.mean(maes))

    print('CV RMSE per fold:', rmses)
    print(f'Mean RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}')

    print('Fitting on full data...')
    model.fit(X, y, cat_features=cat_cols)

    out_model = MODELS / 'model_catboost_final.joblib'
    out_metrics = MODELS / 'model_catboost_final.metrics.json'

    joblib.dump(model, out_model)
    with open(out_metrics, 'w') as f:
        json.dump({'rmse_cv': mean_rmse, 'mae_cv': mean_mae, 'cat_features': cat_cols, 'params': params}, f, indent=2)

    print('Saved final model to', out_model)


if __name__ == '__main__':
    main()
