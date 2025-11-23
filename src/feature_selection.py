"""Selección de features basada en correlación e importancia del modelo.

Funciones para:
- detectar pares de features altamente correlacionadas
- usar importancias de un modelo (LightGBM/estimator con `feature_importances_`) para decidir cuál eliminar
"""
from typing import List, Tuple
import pandas as pd
import numpy as np


def high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.9) -> List[Tuple[str, str, float]]:
    """Devuelve lista de tuplas (col1, col2, corr_value) para pares con |corr| >= threshold.

    Solo considera columnas numéricas del DataFrame.
    """
    df_num = df.select_dtypes(include=[np.number]).copy()
    corr = df_num.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iat[i, j]
            if val >= threshold:
                pairs.append((cols[i], cols[j], float(val)))
    return pairs


def suggest_drops_by_importance(df: pd.DataFrame, model, threshold: float = 0.9) -> List[str]:
    """Sugiere features a eliminar: para cada par altamente correlacionado, elimina
    la columna con menor importancia según `model.feature_importances_`.

    Devuelve lista de columnas sugeridas para drop.
    """
    pairs = high_correlation_pairs(df, threshold=threshold)
    if not pairs:
        return []

    # obtener importances del modelo
    importances = {}
    if hasattr(model, 'feature_importances_'):
        # si el modelo tiene feature_name_, úsalo para mapear
        if hasattr(model, 'feature_name_'):
            fnames = list(model.feature_name_)
        else:
            # intentar inferir de predict input
            fnames = df.select_dtypes(include=[np.number]).columns.tolist()
        for i, f in enumerate(fnames):
            try:
                importances[f] = float(model.feature_importances_[i])
            except Exception:
                importances[f] = 0.0
    else:
        # si no hay importancias, usar varianza como proxy
        for f in df.select_dtypes(include=[np.number]).columns:
            importances[f] = float(df[f].var())

    drops = set()
    for a, b, corrval in pairs:
        ia = importances.get(a, 0.0)
        ib = importances.get(b, 0.0)
        # eliminar la de menor importancia
        if ia <= ib:
            drops.add(a)
        else:
            drops.add(b)

    # retornar solo columnas que existen en df
    return [c for c in list(drops) if c in df.columns]
