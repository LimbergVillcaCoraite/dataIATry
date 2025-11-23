"""Entrenamiento HPO rápido con transformación log1p del target.

Guarda modelo en `models/model_lgbm_log1p_hpo.joblib` si mejora el RMSE medio en escala original.
"""
import json
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

from src.preprocessing import build_features, create_lag_features, prepare_X_y


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_alquiler.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def eval_on_original_scale(estimator, X, y_log, cv):
    """Evalúa RMSE/MAE en escala original usando splits manuales.

    - `y_log` es el target transformado con log1p
    - devuelve mean_rmse, mean_mae (ambos en escala original)
    """
    rmses = []
    maes = []
    for train_idx, test_idx in cv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr_log, y_te_log = y_log.iloc[train_idx], y_log.iloc[test_idx]
        est = estimator.__class__(**estimator.get_params())
        est.fit(X_tr, y_tr_log)
        pred_log = est.predict(X_te)
        # volver a escala original
        y_te = np.expm1(y_te_log)
        pred = np.expm1(pred_log)
        rmses.append(sqrt(mean_squared_error(y_te, pred)))
        maes.append(mean_absolute_error(y_te, pred))
    return float(np.mean(rmses)), float(np.mean(maes))


def run():
    print('Cargando datos...')
    df = pd.read_csv(DATA)

    print('Aplicando features y lags...')
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    print('Preparando X/y (excluyendo u_casuales, u_registrados)...')
    X, y = prepare_X_y(df, target='total_alquileres', exclude=['u_casuales', 'u_registrados'])

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # aplicar log1p al target
    y_log = np.log1p(y)

    tscv = TimeSeriesSplit(n_splits=5)

    param_dist = {
        'n_estimators': [100, 200, 400, 800],
        'num_leaves': [31, 50, 70, 100],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [3, 5, 6, 8, -1],
        'subsample': [0.6, 0.8, 1.0]
    }

    base = LGBMRegressor(objective='regression', verbosity=-1, random_state=42)
    rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=20, cv=tscv,
                            scoring='neg_mean_squared_error', random_state=42, n_jobs=1, verbose=1)

    print('Ejecutando RandomizedSearchCV sobre target log1p...')
    rs.fit(X, y_log)
    print('Best params:', rs.best_params_)

    best = rs.best_estimator_

    print('Evaluando en escala original (CV manual)...')
    mean_rmse, mean_mae = eval_on_original_scale(best, X, pd.Series(y_log, index=y_log.index), tscv)
    print(f'Mean RMSE (original scale): {mean_rmse:.4f}, MAE: {mean_mae:.4f}')

    # comparar con modelo actual
    current_metrics_path = MODELS / 'model_lgbm.metrics.json'
    current_rmse = None
    if current_metrics_path.exists():
        try:
            with open(current_metrics_path, 'r') as f:
                cur = json.load(f)
                current_rmse = float(cur.get('rmse'))
        except Exception:
            current_rmse = None

    print('RMSE actual (model_lgbm):', current_rmse)

    out_model = MODELS / 'model_lgbm_log1p_hpo.joblib'
    out_metrics = MODELS / 'model_lgbm_log1p_hpo.metrics.json'

    if current_rmse is None or mean_rmse < current_rmse:
        print('Mejoró — guardando modelo y métricas...')
        joblib.dump(best, out_model)
        with open(out_metrics, 'w') as f:
            json.dump({'rmse': mean_rmse, 'mae': mean_mae, 'best_params': rs.best_params_}, f, indent=2)
        print('Guardado en', out_model)
    else:
        print('No mejoró; no se guarda.')


if __name__ == '__main__':
    run()
