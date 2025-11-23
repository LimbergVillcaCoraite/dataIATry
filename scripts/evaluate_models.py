"""Evalúa y compara modelos guardados usando TimeSeriesSplit.
Guarda un JSON resumen en `reports/comparison_optuna.json` y un CSV con métricas.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset_alquiler.csv'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

from src.preprocessing import build_features, create_lag_features, prepare_X_y


def evaluate_model(model, X, y, tscv):
    rmses = []
    maes = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        # handle different model types
        try:
            # LightGBM Booster
            if isinstance(model, lgb.Booster):
                preds = model.predict(X_val, num_iteration=getattr(model, 'best_iteration', None))
            else:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
        except Exception:
            # fallback: try predict directly
            preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        maes.append(mean_absolute_error(y_val, preds))
    return float(np.mean(rmses)), float(np.mean(maes))


def load_model_safe(path):
    try:
        return joblib.load(path)
    except Exception:
        # try LightGBM native load
        try:
            return lgb.Booster(model_file=str(path))
        except Exception:
            raise


def main():
    df = pd.read_csv(DATA)
    df = build_features(df)
    df = create_lag_features(df, lags=[1,24,48,168], rolling_windows=[3,6,24,168])
    X, y = prepare_X_y(df, exclude=['u_casuales', 'u_registrados'])
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    tscv = TimeSeriesSplit(n_splits=5)

    candidates = [
        MODELS / 'model_lgbm_optuna.joblib',
        MODELS / 'model_lgbm.joblib',
        MODELS / 'model_catboost_hpo.joblib',
        MODELS / 'model_stacked.joblib',
    ]

    results = {}
    for p in candidates:
        if not p.exists():
            continue
        name = p.stem
        try:
            m = load_model_safe(p)
            rmse, mae = evaluate_model(m, X, y, tscv)
            results[name] = {'rmse': rmse, 'mae': mae, 'path': str(p)}
            print(f'Evaluated {name}: RMSE={rmse:.4f}, MAE={mae:.4f}')
        except Exception as e:
            results[name] = {'error': str(e), 'path': str(p)}
            print(f'Failed to evaluate {name}: {e}')

    out = REPORTS / 'comparison_optuna.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    # also CSV
    df_out = pd.DataFrame.from_dict(results, orient='index')
    csv_out = REPORTS / 'comparison_optuna.csv'
    df_out.to_csv(csv_out)
    print('Saved comparison to', out, csv_out)


if __name__ == '__main__':
    main()
