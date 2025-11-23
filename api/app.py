from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
from pathlib import Path
import pandas as pd
import json
import lightgbm as lgb
import importlib

from src.preprocessing import build_features, create_lag_features, encode_categoricals, get_feature_columns_example

app = FastAPI(title='Alquiler Monopatines - API')


class PredictRequest(BaseModel):
    # aceptar un diccionario de características; las llaves deben coincidir con raw input (fecha, hora, etc.)
    data: dict
    model: Optional[str] = 'model_catboost_hpo'


def _discover_models(models_dir: Path) -> dict:
    """Escanea `models_dir` y devuelve un dict name->path para archivos .joblib

    Esto permite añadir modelos al directorio y que la API los descubra automáticamente.
    """
    out = {}
    for p in models_dir.glob('*.joblib'):
        out[p.stem] = p
    return out


@app.on_event('startup')
def startup_models():
    """Cargar mapa de modelos disponible en memoria (pero no cargarlos todos en RAM).

    Dejamos la carga real para cada petición para ahorrar memoria.
    """
    app.state.models_dir = Path(__file__).resolve().parents[1] / 'models'
    app.state.model_registry = _discover_models(app.state.models_dir)
    # cache para modelos cargados en memoria (evita re-cargar en cada petición)
    app.state.model_cache = {}


# Ensure minimal registry exists even if startup event wasn't executed (useful for tests)
if getattr(app.state, 'model_registry', None) is None:
    app.state.models_dir = Path(__file__).resolve().parents[1] / 'models'
    app.state.model_registry = _discover_models(app.state.models_dir)
    app.state.model_cache = {}


def load_model_path(name: str):
    path = app.state.model_registry.get(name)
    if not path:
        raise HTTPException(status_code=404, detail=f"Modelo {name} no encontrado")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Archivo de modelo {path} no existe")
    return path


@app.post('/predict')
def predict(req: PredictRequest):
    # Preparar DataFrame de entrada
    df = pd.DataFrame([req.data])

    # Aplicar las mismas transformaciones que en entrenamiento (evitar lags en una única fila)
    df = build_features(df)
    if len(df) > 1:
        df = create_lag_features(df)
    # codificar categorías con códigos (consistent con prepare_X_y)
    df = encode_categoricals(df)

    # Alinear columnas con las que el modelo espera. get_feature_columns_example
    # construye las columnas numéricas esperadas por el pipeline.
    expected_cols = get_feature_columns_example(df)
    X = df.select_dtypes(include=['number']).copy()
    # garantizar que el orden y columnas coincidan con lo esperado por el modelo
    X = X.reindex(columns=expected_cols, fill_value=0).fillna(0)

    model_path = load_model_path(req.model)
    # cargar modelo desde cache si está disponible
    model = app.state.model_cache.get(req.model)
    if model is None:
        # intentar joblib.load, si falla y LightGBM está disponible, intentar cargar como Booster
        try:
            model = joblib.load(model_path)
        except Exception:
            try:
                model = lgb.Booster(model_file=str(model_path))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo: {e}")
        app.state.model_cache[req.model] = model

    try:
        # Alinear X con las features que el modelo espera cuando sea posible
        try:
            # CatBoost: feature_names_ , sklearn wrappers: feature_name_
            if hasattr(model, 'feature_names_'):
                feat_names = list(model.feature_names_)
                X = X.reindex(columns=feat_names, fill_value=0)
            elif hasattr(model, 'feature_name_'):
                feat_names = list(model.feature_name_)
                X = X.reindex(columns=feat_names, fill_value=0)
        except Exception:
            pass

        # manejar LightGBM Booster (usa predict con numpy)
        if isinstance(model, lgb.Booster):
            preds = model.predict(X.values)
        else:
            preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

    return {'model': req.model, 'prediction': float(preds[0])}


@app.get('/')
def root():
    """Información básica y modelos disponibles en la API."""
    models = list(app.state.model_registry.keys()) if getattr(app.state, 'model_registry', None) is not None else []
    return {
        'service': 'Alquiler Monopatines - API',
        'available_models': models,
        'predict_endpoint': '/predict (POST)',
        'models_endpoint': '/models (GET)'
    }


@app.get('/models')
def list_models():
    """Devuelve modelos disponibles junto a su ruta y métricas si existen."""
    out = {}
    for name, path in app.state.model_registry.items():
        info = {'path': str(path)}
        metrics_file = path.with_suffix('.metrics.json')
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as fh:
                    info['metrics'] = json.load(fh)
            except Exception:
                info['metrics'] = 'invalid json'
        out[name] = info
    return out
