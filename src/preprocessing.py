"""Preprocesamiento y generación de features para el proyecto.

Funciones principales:
- build_features(df): añade variables de tiempo y limpiezas básicas.
- create_lag_features(df, col, lags): crea rezagos temporales del target.
- encode_categoricals(df): encodeo sencillo para categoricals.
- prepare_X_y(df, target, drop_cols): prepara X e y listos para el modelo.
- get_feature_columns_example(): devuelve lista de columnas esperadas (útil para la API).

Este módulo está escrito para ser compacto y robusto: evita dependencias
pesadas y maneja NA/inf apropiadamente.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Asegurar que fecha existe y es datetime
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    else:
        raise KeyError("Se requiere la columna 'fecha' en el dataframe")

    # Variables temporales
    df["hour"] = df["fecha"].dt.hour
    df["dayofweek"] = df["fecha"].dt.dayofweek
    df["month"] = df["fecha"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Orden por fecha
    df = df.sort_values("fecha").reset_index(drop=True)

    return df


def create_lag_features(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    df = df.copy()
    for l in lags:
        name = f"{col}_lag_{l}"
        df[name] = df[col].shift(l)
    return df


def encode_categoricals(df: pd.DataFrame, cols: List[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    if cols is None:
        # detectar categorías simples (object o category)
        cols = [c for c, t in df.dtypes.items() if t == "object"]
    for c in cols:
        # label encoding simple que mantiene orden y evita dependencias
        df[c] = df[c].astype("category").cat.codes
    return df


def prepare_X_y(
    df: pd.DataFrame,
    target: str = "total_alquileres",
    drop_cols: List[str] | None = None,
    lags: List[int] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if drop_cols is None:
        drop_cols = ["u_casuales", "u_registrados"]

    # construir features temporales
    df = build_features(df)

    # añadir rezagos del target si se pide
    if lags is None:
        lags = [1, 24, 168]
    if target in df.columns:
        df = create_lag_features(df, target, lags)

    # eliminar columnas que no deben ser usadas
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Valores infinitos a NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # seleccionar columnas candidatas para X
    # excluir fecha y target
    exclude = {"fecha", target}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    # encode simples
    X = encode_categoricals(X)

    # Alineamiento: quitar filas con NA en X o y
    if target in df.columns:
        y = df[target].copy()
    else:
        y = pd.Series([np.nan] * len(df), index=df.index)

    # concatenar para eliminar filas con NA en cualquiera
    both = pd.concat([X, y.rename("__y")], axis=1)
    both = both.dropna()

    X_clean = both.drop(columns=["__y"]).reset_index(drop=True)
    y_clean = both["__y"].reset_index(drop=True)

    return X_clean, y_clean


def get_feature_columns_example() -> List[str]:
    # lista mínima de columnas que el modelo espera (usada por la API para alinear)
    base = ["hour", "dayofweek", "month", "is_weekend"]
    lags = ["total_alquileres_lag_1", "total_alquileres_lag_24", "total_alquileres_lag_168"]
    return base + lags
