"""Preprocesamiento modular y reproducible.

Este módulo encapsula transformaciones reproducibles que se deben aplicar tanto
en entrenamiento como en producción (API). La idea es mantener funciones pequeñas
y documentadas que puedan combinarse en un pipeline.
"""
from typing import Tuple
import pandas as pd
import numpy as np

from src.data import basic_clean


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica feature engineering simple y reproducible.

    - Extrae componentes temporales de `fecha` si existe.
    - Asegura que `hora` sea entera.
    - Crea variables cíclicas si es útil (hora -> sin/cos) para capturar periodicidad.
    """
    df = df.copy()

    # Aplicar limpieza básica primero
    df = basic_clean(df)

    # Extraer componentes temporales solo si `fecha` se convirtió correctamente
    if 'fecha' in df.columns and pd.api.types.is_datetime64_any_dtype(df['fecha']):
        df['year'] = df['fecha'].dt.year
        df['month'] = df['fecha'].dt.month
        df['day'] = df['fecha'].dt.day
        df['dayofweek'] = df['fecha'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    # Hora: asegurar entero y generar representación cíclica
    if 'hora' in df.columns:
        df['hora'] = df['hora'].astype('Int64')
        # Representación cíclica: sin/cos para hora (0-23)
        try:
            hour = df['hora'].fillna(0).astype(int)
            df['hora_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hora_cos'] = np.cos(2 * np.pi * hour / 24)
        except Exception:
            # En caso de fallo, dejar sin/cos con NaN
            df['hora_sin'] = np.nan
            df['hora_cos'] = np.nan

    # Podríamos codificar `clima` como categórica ordinal si es numérico
    if 'clima' in df.columns:
        # convertir a tipo categórico para reducir memoria y permitir códigos ordinales
        try:
            df['clima'] = df['clima'].astype('category')
        except Exception:
            pass

    # Convertir columnas de baja cardinalidad a 'category' para optimizar memoria
    for col in ['temporada', 'feriado', 'dia_semana']:
        if col in df.columns:
            try:
                df[col] = df[col].astype('category')
            except Exception:
                continue

    return df


def create_lag_features(df: pd.DataFrame, lags: list = [1, 24], rolling_windows: list = [24]) -> pd.DataFrame:
    """Crea features de series temporales (lags y medias móviles).

    - `lags`: lista de enteros que representan desplazamientos en horas (ej. 1, 24).
    - `rolling_windows`: ventanas para medias móviles (en horas).

    Motivo: lag features aportan información de dependencia temporal inmediata y estacional.
    """
    df = df.copy()
    # Necesitamos un índice temporal
    if 'fecha' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['fecha']):
        # intentar convertir
        df['fecha'] = pd.to_datetime(df.get('fecha', None), errors='coerce')

    # ordenar por fecha para que los lags sean consistentes
    if 'fecha' in df.columns:
        df = df.sort_values('fecha').reset_index(drop=True)

    if 'total_alquileres' in df.columns:
        for lag in lags:
            df[f'lag_{lag}'] = df['total_alquileres'].shift(lag)
        for w in rolling_windows:
            df[f'rolling_mean_{w}'] = df['total_alquileres'].shift(1).rolling(window=w, min_periods=1).mean()

    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list = ['clima', 'temporada']) -> pd.DataFrame:
    """Codifica variables categóricas con one-hot (limitando cardinalidad si es necesario).

    - Guardamos prefijos claros para facilitar depuración y despliegue.
    """
    df = df.copy()
    # Para mejorar rendimiento y evitar expansión excesiva de columnas usamos códigos
    # ordinales para categorías (LightGBM maneja bien enteros). Esto mantiene la
    # dimensionalidad baja y acelera tanto entrenamiento como predicción.
    for col in cat_cols:
        if col in df.columns:
            try:
                # convertir en categórico si no lo es
                if not pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].astype('category')
                # reemplazar por códigos (enteros) y renombrar la columna para claridad
                df[f'{col}_cat'] = df[col].cat.codes
                df = df.drop(columns=[col])
            except Exception:
                # si falla, dejar la columna tal cual
                continue
    return df


def prepare_X_y(df: pd.DataFrame, target: str = 'total_alquileres', exclude: list | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Construye la matriz de características X y el vector objetivo y.

    - `exclude` permite quitar columnas no deseadas (p.ej. `u_casuales`, `u_registrados`).
    - Se hace imputación numérica simple (mediana) para variables numéricas.
    - Devuelve X (solo columnas numéricas por simplicidad) e y.
    """
    if exclude is None:
        exclude = []
    df = df.copy()

    # Asegurar que el objetivo existe
    if target not in df.columns:
        raise ValueError(f"Objetivo '{target}' no encontrado en el DataFrame")

    y = df[target]

    # Excluir columnas irrelevantes
    to_drop = [target] + exclude
    X = df.drop(columns=[c for c in to_drop if c in df.columns])

    # Codificar categóricas antes de seleccionar numéricas
    X = encode_categoricals(X)

    # Mantener solo columnas numéricas para un pipeline baseline
    X_num = X.select_dtypes(include=[np.number]).copy()

    # Imputación simple: mediana
    X_num = X_num.fillna(X_num.median())

    return X_num, y


def get_feature_columns_example(df: pd.DataFrame) -> list:
    """Función auxiliar para devolver columnas que el modelo espera.

    Útil para la API: tomar un DataFrame con una fila y alinear columnas.
    """
    df2 = build_features(df.head(1))
    X_num, _ = prepare_X_y(df2, exclude=['u_casuales', 'u_registrados'])
    return list(X_num.columns)
