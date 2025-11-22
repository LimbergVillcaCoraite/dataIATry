import pandas as pd
import numpy as np
from pathlib import Path


# Este módulo se encarga exclusivamente de funciones de carga y limpieza básica.
# Comentarios colocados en cada paso explican el motivo de la transformación.


def load_data(csv_path: str | Path):
    """Carga un CSV y devuelve un DataFrame.

    Notas:
    - `low_memory=False` para que pandas infiera tipos más consistentemente.
    - No se hace conversión de tipos aquí: eso se delega en `basic_clean`.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza básica y correcciones:

    Pasos importantes:
    - Reconstruye `total_alquileres` a partir de `u_casuales` + `u_registrados` si falta.
      Motivo: el objetivo es `total_alquileres`; cuando falta es razonable reconstruirlo
      si las dos sumandas están presentes (evita pérdida de filas).
    - Convierte `fecha` a datetime para permitir splits temporales y extracción de features.
    - Normaliza tipos enteros que pandas infirió como float.
    - Detecta y trata outliers en `sensacion_termica` y `velocidad_viento`.

    Notas sobre outliers:
    - Si existe una diferencia muy grande entre el max y el percentil 99, hacemos clipping
      al p99 y añadimos una columna flag `*_outlier_clipped` para rastrear qué registros
      fueron recortados (esto ayuda en análisis posteriores).
    - Si la mediana es muy pequeña (p.ej. valores esperados en [0,1]) pero aparecen
      valores > 50, los consideramos sospechosos y los marcamos como NA (se añade flag).
    - También se forzan a NaN valores absurdamente grandes (>1000) por seguridad.
    """

    # Reparar total_alquileres si falta y si existen los componentes
    if 'total_alquileres' in df.columns and ('u_casuales' in df.columns and 'u_registrados' in df.columns):
        mask = df['total_alquileres'].isna()
        if mask.any():
            df.loc[mask, 'total_alquileres'] = (
                df.loc[mask, 'u_casuales'].fillna(0) + df.loc[mask, 'u_registrados'].fillna(0)
            )

    # Convertir fecha a datetime (errores -> NaT)
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

    # Corregir columnas que deberían ser enteras
    for col in ['hora', 'dia_semana', 'mes', 'anio', 'temporada', 'feriado', 'dia_trabajo', 'clima']:
        if col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                # Si todos los valores no nulos son enteros, convertir a Int64 (soporta NA)
                if (df[col].dropna() % 1 == 0).all():
                    df[col] = df[col].astype('Int64')

    # Tratamiento heurístico de outliers para columnas climáticas
    for col in ['sensacion_termica', 'velocidad_viento']:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            vmax = vals.max()
            vmedian = vals.median()
            p99 = vals.quantile(0.99)

            # Si el máximo es mucho mayor que el p99, recortamos al p99 y marcamos los recortes
            if p99 > 0 and vmax > 10 * p99:
                df[col + '_outlier_clipped'] = False
                clip_val = p99
                mask_out = df[col] > clip_val
                df.loc[mask_out, col + '_outlier_clipped'] = True
                df.loc[mask_out, col] = clip_val

            # Si aparecen valores absurdamente grandes, forzarlos a NaN
            df.loc[df[col] > 1000, col] = np.nan

            # Si la mediana es pequeña (valores típicos en [0,1]) pero existen valores >50,
            # marcarlos como NaN y añadir flag
            if vmedian < 1 and vmax > 50:
                df[col + '_outlier_large'] = False
                mask_large = df[col] > 50
                if mask_large.any():
                    df.loc[mask_large, col + '_outlier_large'] = True
                    df.loc[mask_large, col] = np.nan

    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'fecha' in out.columns:
        out['year'] = out['fecha'].dt.year
        out['month'] = out['fecha'].dt.month
        out['day'] = out['fecha'].dt.day
        out['dayofweek'] = out['fecha'].dt.dayofweek
        out['is_weekend'] = out['dayofweek'].isin([5,6]).astype(int)
    # asegurar hora como entero
    if 'hora' in out.columns:
        out['hora'] = out['hora'].astype('Int64')

    return out


def get_feature_matrix(df: pd.DataFrame, exclude: list | None = None):
    if exclude is None:
        exclude = []
    X = df.copy()
    # columnas objetivo y excluidas
    for c in exclude:
        if c in X.columns:
            X = X.drop(columns=[c])
    return X
