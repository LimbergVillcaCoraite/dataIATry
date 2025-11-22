"""Entrenamiento, evaluación y guardado de modelos.

Este módulo centraliza las rutinas para entrenar modelos, calcular métricas y
guardar artefactos (modelos y métricas). Mantener esta lógica separada facilita
probar distintos algoritmos sin duplicar código.
"""
from pathlib import Path
from typing import Any
import json
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid


def evaluate(y_true, y_pred):
    """Calcula métricas RMSE y MAE.

    Devolvemos un diccionario simple con valores float.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {'rmse': rmse, 'mae': mae}


def train_random_forest(X_train, y_train, **kwargs) -> Any:
    """Entrena un RandomForestRegressor con kwargs y devuelve el modelo.
    Se usan parámetros por defecto razonables para un baseline.
    """
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, n_estimators: int = 500, learning_rate: float = 0.05, **kwargs) -> Any:
    """Entrena un modelo LightGBM regresor.

    Usamos la API sklearn de LightGBM para integración sencilla.
    Parámetros explícitos evitan pasar dos veces `n_estimators` si se incluye
    en `kwargs` desde tests o llamadas externas.
    """
    params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
    }
    # `kwargs` puede sobreescribir parámetros explícitos si es necesario
    params.update(kwargs)
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def simple_param_search(model_type: str, X_train, y_train, X_val, y_val):
    """Búsqueda de hiperparámetros simple sobre un grid pequeño.

    - `model_type` = 'rf' o 'lgbm'
    - Retorna el mejor modelo entrenado y sus métricas.
    """
    best_model = None
    best_metrics = {'rmse': float('inf')}
    best_params = None

    if model_type == 'rf':
        # Para acelerar la búsqueda usamos un grid pequeño y entrenamos sobre una muestra
        grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, None],
        }
        # usar muestra si el dataset es grande (reducimos frac para acelerar)
        if len(X_train) > 5000:
            X_search = X_train.sample(frac=0.2, random_state=42)
            y_search = y_train.loc[X_search.index]
        else:
            X_search = X_train
            y_search = y_train

        for params in ParameterGrid(grid):
            # Durante la búsqueda usamos un único job para evitar sobrecarga de joblib
            m = RandomForestRegressor(n_jobs=1, random_state=42, **params)
            m.fit(X_search, y_search)
            preds = m.predict(X_val)
            metrics = evaluate(y_val, preds)
            if metrics['rmse'] < best_metrics['rmse']:
                best_metrics = metrics
                best_model = m
                best_params = params

    elif model_type == 'lgbm':
        grid = {
            'n_estimators': [100, 300],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 63],
        }
        # usar muestra si el dataset es grande (reducimos frac para acelerar)
        if len(X_train) > 5000:
            X_search = X_train.sample(frac=0.2, random_state=42)
            y_search = y_train.loc[X_search.index]
        else:
            X_search = X_train
            y_search = y_train

        for params in ParameterGrid(grid):
            # Usar n_jobs=1 para la búsqueda para reducir overhead
            m = lgb.LGBMRegressor(random_state=42, n_jobs=1, **params)
            m.fit(X_search, y_search)
            preds = m.predict(X_val)
            metrics = evaluate(y_val, preds)
            if metrics['rmse'] < best_metrics['rmse']:
                best_metrics = metrics
                best_model = m
                best_params = params

    # Si encontramos parámetros, reentrenar el mejor modelo con todo el X_train (no sólo la muestra)
    if best_params is not None:
        if model_type == 'rf':
            final_model = RandomForestRegressor(n_jobs=-1, random_state=42, **best_params)
            final_model.fit(X_train, y_train)
        else:
            final_model = lgb.LGBMRegressor(n_jobs=-1, random_state=42, **best_params)
            final_model.fit(X_train, y_train)
        return final_model, best_metrics

    return best_model, best_metrics


def save_model_and_metrics(model: Any, model_path: Path, metrics: dict):
    """Guarda el modelo (joblib) y las métricas (JSON) junto a él.

    Estructura:
      models/<name>.joblib
      models/<name>.metrics.json
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path = model_path.with_suffix('.metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
