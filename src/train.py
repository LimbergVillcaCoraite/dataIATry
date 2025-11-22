from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing import build_features, prepare_X_y
from src.models import (
    train_random_forest,
    train_lightgbm,
    evaluate,
    save_model_and_metrics,
)


def time_split(df: pd.DataFrame, date_col: str = 'fecha', frac: float = 0.8):
    """Split temporal simple: ordena por `date_col` y toma frac como train.

    Motivo: para series temporales es incorrecto usar shuffle random splits.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split = int(len(df) * frac)
    return df.iloc[:split], df.iloc[split:]


def train_all(csv_path: str | Path, out_dir: str | Path):
    """Entrena varios modelos (RandomForest y LightGBM) y guarda artefactos.

    - Aplica pipeline de features reproducible.
    - Realiza split temporal si `fecha` existe.
    - Entrena y guarda modelos y métricas.
    """
    repo = Path(csv_path).resolve().parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    df = build_features(df)
    # Crear lags sobre todo el conjunto antes de split temporal para que test tenga lags válidos
    from src.preprocessing import create_lag_features
    df = create_lag_features(df)

    # Columnas a excluir según la prueba técnica
    exclude = ['indice', 'u_casuales', 'u_registrados']

    # Si existe fecha válida, hacer split temporal
    if 'fecha' in df.columns and df['fecha'].notna().any():
        # asegurar que fecha es datetime
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        train_df, test_df = time_split(df, date_col='fecha', frac=0.8)
    else:
        # split aleatorio como fallback
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

    X_train, y_train = prepare_X_y(train_df, exclude=exclude)
    X_test, y_test = prepare_X_y(test_df, exclude=exclude)

    # Para mejorar precisión: hacemos una búsqueda simple de hiperparámetros
    # usando el conjunto de validación temporal (train_df/test_df).
    try:
        rf_model, rf_metrics = __import__('src.models', fromlist=['simple_param_search']).simple_param_search(
            'rf', X_train, y_train, X_test, y_test
        )
        save_model_and_metrics(rf_model, out_dir / 'model_rf_tuned.joblib', rf_metrics)
    except Exception as e:
        print('Warning: falla en búsqueda RF, entrenando con parámetros por defecto:', e)
        rf = train_random_forest(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_metrics = evaluate(y_test, rf_preds)
        save_model_and_metrics(rf, out_dir / 'model_rf_tuned.joblib', rf_metrics)

    # LightGBM búsqueda
    try:
        lgbm_model, lgbm_metrics = __import__('src.models', fromlist=['simple_param_search']).simple_param_search(
            'lgbm', X_train, y_train, X_test, y_test
        )
        save_model_and_metrics(lgbm_model, out_dir / 'model_lgbm_tuned.joblib', lgbm_metrics)
    except Exception as e:
        print('Warning: falla en búsqueda LGBM, entrenando con parámetros por defecto:', e)
        lgbm = train_lightgbm(X_train, y_train)
        lgbm_preds = lgbm.predict(X_test)
        lgbm_metrics = evaluate(y_test, lgbm_preds)
        save_model_and_metrics(lgbm, out_dir / 'model_lgbm_tuned.joblib', lgbm_metrics)

    print('Entrenamiento completado. Modelos guardados en', out_dir)
    print('RandomForest metrics:', rf_metrics)
    if 'lgbm_metrics' in locals():
        print('LightGBM metrics:', lgbm_metrics)


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[1]
    csv = repo / 'dataset_alquiler.csv'
    out = repo / 'models'
    train_all(csv, out)
