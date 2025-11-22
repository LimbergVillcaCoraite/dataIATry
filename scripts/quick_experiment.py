"""Experimento rápido: agregar lags/rolling y hacer RandomizedSearchCV sobre LightGBM.

Guarda el modelo en `models/model_lgbm_quick_hpo.joblib` si mejora el RMSE medio.
"""
import json
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

from src.preprocessing import build_features, create_lag_features, prepare_X_y


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_alquiler.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def run_quick_hpo():
    print("Cargando datos...")
    df = pd.read_csv(DATA)

    print("Aplicando features (lags y rolling windows)...")
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    print("Preparando X/y (excluyendo u_casuales, u_registrados)...")
    X, y = prepare_X_y(df, target='total_alquileres', exclude=['u_casuales', 'u_registrados'])

    # Drop rows with NaN in y or features (first rows because of lag)
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Parametrización para búsqueda aleatoria
    param_dist = {
        'n_estimators': [100, 200, 400, 800],
        'num_leaves': [31, 50, 70, 100],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [3, 5, 6, 8, -1],
        'subsample': [0.6, 0.8, 1.0]
    }

    print("Lanzando RandomizedSearchCV (n_iter=20, cv=TimeSeriesSplit)...")
    base = LGBMRegressor(objective='regression', verbosity=-1, random_state=42)
    rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=20, cv=tscv,
                            scoring='neg_mean_squared_error', random_state=42, n_jobs=1, verbose=1)

    rs.fit(X, y)

    print("Mejores parámetros:", rs.best_params_)

    # Evaluación cross-val con best_estimator
    best = rs.best_estimator_
    scores = cross_val_score(best, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1)
    rmses = [sqrt(-s) for s in scores]
    mean_rmse = float(np.mean(rmses))
    # MAE usando cross_val_score con scoring neg_mean_absolute_error
    mae_scores = cross_val_score(best, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
    mae = float(-np.mean(mae_scores))

    print(f"CV RMSE per fold: {rmses}")
    print(f"Mean RMSE: {mean_rmse:.4f}, MAE: {mae:.4f}")

    # Leer métricas actuales del LGBM base (si existen) para comparar
    current_metrics_path = MODELS / 'model_lgbm.metrics.json'
    current_rmse = None
    if current_metrics_path.exists():
        try:
            with open(current_metrics_path, 'r') as f:
                cur = json.load(f)
                current_rmse = float(cur.get('rmse'))
        except Exception:
            current_rmse = None

    print(f"RMSE actual (model_lgbm): {current_rmse}")

    # Si mejora, guardar
    out_model_path = MODELS / 'model_lgbm_quick_hpo.joblib'
    out_metrics_path = MODELS / 'model_lgbm_quick_hpo.metrics.json'

    if current_rmse is None or mean_rmse < current_rmse:
        print("Mejoró el RMSE. Guardando modelo y métricas...")
        joblib.dump(best, out_model_path)
        metrics = {'rmse': mean_rmse, 'mae': mae, 'best_params': rs.best_params_}
        with open(out_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Guardado en: {out_model_path}")
    else:
        print("No mejoró respecto al modelo actual. No se guarda.")


if __name__ == '__main__':
    run_quick_hpo()
