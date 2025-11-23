"""HPO rápido para CatBoost usando TimeSeriesSplit y RandomizedSearchCV.

Guarda `models/model_catboost_hpo.joblib` y `models/model_catboost_hpo.metrics.json` si mejora RMSE.
"""
from pathlib import Path
import json
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error

try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise

from src.preprocessing import build_features, create_lag_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_alquiler.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def prepare_catboost_data(df, target='total_alquileres', exclude=None):
    if exclude is None:
        exclude = []
    # build features and lags
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    # drop rows without target
    df = df.loc[~df[target].isna()].copy()

    X = df.drop(columns=[c for c in ([target] + exclude) if c in df.columns])
    y = df[target]

    # Convert object -> category for CatBoost
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')

    # Ensure category dtypes where appropriate
    cat_cols = list(X.select_dtypes(include=['category']).columns)

    # Simple numeric imputation
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.fillna(X_num.median())
    # For categorical columns, fillna with a new category
    for c in cat_cols:
        X[c] = X[c].cat.add_categories(['__missing__']).fillna('__missing__')

    # Recombine preserving dtypes
    X_final = pd.concat([X_num, X[cat_cols]], axis=1)

    return X_final, y, cat_cols


def run():
    print('Cargando datos...')
    df = pd.read_csv(DATA)

    print('Preparando dataset para CatBoost...')
    X, y, cat_cols = prepare_catboost_data(df, exclude=['u_casuales', 'u_registrados'])

    print('Categorical columns:', cat_cols)

    tscv = TimeSeriesSplit(n_splits=5)

    param_dist = {
        'iterations': [200, 400, 800, 1200],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }

    base = CatBoostRegressor(loss_function='RMSE', verbose=0, random_seed=42)

    rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=40, cv=tscv,
                            scoring='neg_mean_squared_error', random_state=42, n_jobs=1, verbose=1)

    print('Ejecutando RandomizedSearchCV (CatBoost)...')
    # pass cat_features via fit
    rs.fit(X, y, cat_features=cat_cols)

    print('Mejores parámetros:', rs.best_params_)

    best = rs.best_estimator_

    # evaluar CV RMSE manualmente (para pasar cat_features en cada fit)
    rmses = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        est = CatBoostRegressor(**best.get_params())
        est.fit(X_tr, y_tr, cat_features=cat_cols, verbose=0)
        pred = est.predict(X_te)
        rmses.append(sqrt(mean_squared_error(y_te, pred)))
    mean_rmse = float(np.mean(rmses))

    print('CV RMSE per fold:', rmses)
    print('Mean RMSE:', mean_rmse)

    # leer rmse actual
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

    out_model = MODELS / 'model_catboost_hpo.joblib'
    out_metrics = MODELS / 'model_catboost_hpo.metrics.json'

    if current_rmse is None or mean_rmse < current_rmse:
        print('Mejoró — guardando modelo y métricas...')
        joblib.dump(best, out_model)
        with open(out_metrics, 'w') as f:
            json.dump({'rmse': mean_rmse, 'best_params': rs.best_params_, 'cat_features': cat_cols}, f, indent=2)
        print('Guardado en', out_model)
    else:
        print('No mejoró; no se guarda.')


if __name__ == '__main__':
    run()
"""HPO rápido para CatBoost usando TimeSeriesSplit.

Guarda el modelo en `models/model_catboost_hpo.joblib` si mejora el RMSE del LGBM actual.
"""
import json
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error

try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise

from src.preprocessing import build_features, create_lag_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_alquiler.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def run():
    print('Cargando datos...')
    df = pd.read_csv(DATA)

    print('Aplicando features y lags...')
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    target = 'total_alquileres'
    exclude = ['u_casuales', 'u_registrados']

    if target not in df.columns:
        raise RuntimeError('Objetivo no encontrado')

    y = df[target]

    # Construir X antes de codificar categorías para que CatBoost reciba cat_features
    X_raw = df.drop(columns=[c for c in [target] + exclude if c in df.columns])

    # quitar filas donde objetivo es NaN (lags iniciales)
    mask = ~y.isna()
    X_raw = X_raw.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Identificar features categóricas (dtype 'category' generado por build_features)
    cat_cols = list(X_raw.select_dtypes(include=['category']).columns)
    print('Categorical columns detected for CatBoost:', cat_cols)

    # Para CatBoost, pasar DataFrame tal cual;
    # rellenar numéricos con medianas y convertir NaN en categóricas a string
    X = X_raw.copy()
    X_num = X.select_dtypes(include=[np.number])
    X[X_num.columns] = X_num.fillna(X_num.median())
    # rellenar categorías faltantes con un token explícito
    for c in cat_cols:
        try:
            X[c] = X[c].astype(object).fillna('__MISSING__')
        except Exception:
            X[c] = X[c].fillna('__MISSING__')

    tscv = TimeSeriesSplit(n_splits=5)

    param_dist = {
        'iterations': [200, 400, 800],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0]
    }

    model = CatBoostRegressor(loss_function='RMSE', verbose=0, random_seed=42)

    rs = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=40, cv=tscv,
                            scoring='neg_mean_squared_error', random_state=42, n_jobs=1, verbose=1)

    print('Ejecutando RandomizedSearchCV para CatBoost...')
    # pasar cat_features durante fit
    rs.fit(X, y, cat_features=cat_cols)

    print('Mejores parámetros:', rs.best_params_)

    best = rs.best_estimator_

    # Evaluación CV (RMSE y MAE)
    scores = cross_val_score(best, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1, fit_params={'cat_features': cat_cols})
    rmses = [sqrt(-s) for s in scores]
    mean_rmse = float(np.mean(rmses))

    mae_scores = cross_val_score(best, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1, fit_params={'cat_features': cat_cols})
    mean_mae = float(-np.mean(mae_scores))

    print(f'CV RMSE per fold: {rmses}')
    print(f'Mean RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}')

    # comparar con LGBM actual
    curr_path = MODELS / 'model_lgbm.metrics.json'
    current_rmse = None
    if curr_path.exists():
        with open(curr_path, 'r') as f:
            cur = json.load(f)
            current_rmse = float(cur.get('rmse'))

    print('RMSE actual (model_lgbm):', current_rmse)

    out_model = MODELS / 'model_catboost_hpo.joblib'
    out_metrics = MODELS / 'model_catboost_hpo.metrics.json'

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
