"""Valida un modelo CatBoost guardado.

Comprueba:
- que el archivo existe
- que el modelo tiene `feature_names_` y `get_cat_feature_indices()`
- que una fila de ejemplo genera una predicción sin lanzar error
"""
from pathlib import Path
import joblib
import pandas as pd
import json
from src.preprocessing import build_features, create_lag_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model_catboost_final.joblib'


def main():
    if not MODEL_PATH.exists():
        print('Modelo final no encontrado en', MODEL_PATH)
        return 2

    model = joblib.load(MODEL_PATH)
    print('Loaded model type:', type(model))

    has_feat = hasattr(model, 'feature_names_') or hasattr(model, 'feature_name_')
    print('Has feature names:', has_feat)
    if hasattr(model, 'feature_names_'):
        print('n features:', len(model.feature_names_))
        print('first 10 features:', model.feature_names_[:10])

    if hasattr(model, 'get_cat_feature_indices'):
        try:
            cat_idx = model.get_cat_feature_indices()
            print('cat feature indices:', cat_idx)
        except Exception as e:
            print('Error al obtener cat indices:', e)

    # try to predict with a sample row from dataset
    # Create a sample row following the same preprocessing used for training
    df = pd.read_csv(ROOT / 'dataset_alquiler.csv')
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    # drop rows without target (as training did)
    if 'total_alquileres' in df.columns:
        mask = ~df['total_alquileres'].isna()
        df = df.loc[mask].reset_index(drop=True)

    X = df.drop(columns=[c for c in ['total_alquileres', 'u_casuales', 'u_registrados'] if c in df.columns])
    # drop fecha raw
    if 'fecha' in X.columns:
        X = X.drop(columns=['fecha'])

    # numeric imputation
    import numpy as np
    X_num = X.select_dtypes(include=[np.number]).copy()
    if not X_num.empty:
        X[X_num.columns] = X_num.fillna(X_num.median())

    # categorical handling: convert categories to string and fill missing
    cat_cols = list(X.select_dtypes(include=['category']).columns)
    for c in cat_cols:
        try:
            X[c] = X[c].astype(object).fillna('__MISSING__')
        except Exception:
            X[c] = X[c].fillna('__MISSING__')

    row = X.head(1)
    try:
        pred = model.predict(row)
        print('Prediction OK, value:', pred)
    except Exception as e:
        print('Prediction raised error:', e)
        return 3

    # save a small report
    rpt = ROOT / 'reports' / 'validation_model_catboost.json'
    rpt.parent.mkdir(exist_ok=True, parents=True)
    with open(rpt, 'w') as f:
        json.dump({'model': str(MODEL_PATH), 'has_feature_names': has_feat}, f, indent=2)

    print('Validation report saved to', rpt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
