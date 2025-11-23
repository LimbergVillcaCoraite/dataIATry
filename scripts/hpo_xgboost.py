"""HPO rápido para XGBoost usando TimeSeriesSplit y RandomizedSearchCV.

Guarda `models/model_xgb_hpo.joblib` y `models/model_xgb_hpo.metrics.json`.
"""
from pathlib import Path
import json
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
except Exception:
    raise RuntimeError('xgboost is required to run this script')

from src.preprocessing import build_features, create_lag_features, prepare_X_y


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_alquiler.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def run(n_iter=40):
    print('Cargando datos...')
    df = pd.read_csv(DATA)

    print('Aplicando features y lags...')
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    print('Preparando X/y...')
    X, y = prepare_X_y(df, exclude=['u_casuales', 'u_registrados'])

    # XGBoost requiere tipos numéricos / categoricos; eliminar columnas datetime
    if 'fecha' in X.columns:
        print("Eliminando columna 'fecha' (datetime) antes de ajustar XGBoost")
        X = X.drop(columns=['fecha'])

    print('Shape X,y:', X.shape, y.shape)

    tscv = TimeSeriesSplit(n_splits=5)

    param_dist = {
        'n_estimators': [100, 200, 400, 800],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1, 5],
        'reg_lambda': [1, 3, 5]
    }

    base = XGBRegressor(objective='reg:squarederror', tree_method='hist', eval_metric='rmse', random_state=42, n_jobs=4)

    rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=n_iter, cv=tscv,
                            scoring='neg_mean_squared_error', random_state=42, n_jobs=1, verbose=2)

    print('Ejecutando RandomizedSearchCV (XGBoost)...')
    try:
        rs.fit(X, y)
    except Exception as e:
        print('Error durante RandomizedSearchCV:', e)
        raise

    print('Mejores parámetros:', rs.best_params_)

    best = rs.best_estimator_

    # evaluar CV RMSE/MAE manualmente
    rmses = []
    maes = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        est = XGBRegressor(**best.get_params())
        est.fit(X_tr, y_tr)
        pred = est.predict(X_te)
        rmses.append(sqrt(mean_squared_error(y_te, pred)))
        maes.append(mean_absolute_error(y_te, pred))

    mean_rmse = float(np.mean(rmses))
    mean_mae = float(np.mean(maes))

    print('CV RMSE per fold:', rmses)
    print('Mean RMSE:', mean_rmse, 'MAE:', mean_mae)

    out_model = MODELS / 'model_xgb_hpo.joblib'
    out_metrics = MODELS / 'model_xgb_hpo.metrics.json'

    print('Guardando modelo y métricas...')
    joblib.dump(best, out_model)
    with open(out_metrics, 'w') as f:
        json.dump({'rmse': mean_rmse, 'mae': mean_mae, 'best_params': rs.best_params_}, f, indent=2)

    print('Guardado en', out_model)


if __name__ == '__main__':
    run(n_iter=40)
