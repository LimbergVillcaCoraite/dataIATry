#!/usr/bin/env python3
"""Ligera búsqueda de hiperparámetros para LightGBM usando TimeSeriesSplit.

Este script hace una búsqueda limitada (RandomizedSearchCV) y guarda el mejor modelo
en `models/model_lgbm_hpo.joblib` junto con un archivo de métricas JSON.
"""
import json
from pathlib import Path
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

from src.data import load_data, basic_clean
from src.preprocessing import build_features, prepare_X_y


def main():
    root = Path('.')
    data_path = root / 'dataset_alquiler.csv'
    out_dir = root / 'models'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Cargando datos...')
    df = load_data(data_path)
    df = basic_clean(df)
    df_feat = build_features(df)
    # Crear lags antes del split para evitar leakage y asegurar que test tenga valores de lag
    from src.preprocessing import create_lag_features
    df_feat = create_lag_features(df_feat)

    X, y = prepare_X_y(df_feat, target='total_alquileres')

    # temporal split: 80% train, 20% test
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print('Preparando búsqueda HPO (limitada)...')
    param_dist = {
        'num_leaves': [31, 50, 70, 100],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'n_estimators': [100, 200, 400, 800],
        'max_depth': [-1, 6, 10, 20],
        'subsample': [0.6, 0.8, 1.0]
    }

    model = lgb.LGBMRegressor(random_state=42, n_jobs=1)
    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=24,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        random_state=42,
        n_jobs=1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best = search.best_estimator_
    print('Mejor params:', search.best_params_)

    # retrain best on full training partition
    best.fit(X_train, y_train)

    preds = best.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = mean_absolute_error(y_test, preds)

    out_model = out_dir / 'model_lgbm_hpo.joblib'
    joblib.dump(best, out_model)

    metrics = {'rmse': float(rmse), 'mae': float(mae), 'best_params': search.best_params_}
    with open(out_dir / 'model_lgbm_hpo.metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('HPO completado. Modelo guardado en', out_model)
    print('Métricas:', metrics)


if __name__ == '__main__':
    main()
