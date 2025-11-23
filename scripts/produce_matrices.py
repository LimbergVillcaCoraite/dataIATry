"""Genera y guarda: 1) matriz de correlación de features numéricos, 2) 'matriz de confusión'
para regresión creada binning por cuantiles (y_true vs y_pred).

Guarda resultados en `reports/figs/`.
"""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.preprocessing import build_features, create_lag_features, prepare_X_y


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_alquiler.csv"
FIGS = ROOT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def plot_corr_matrix(df_num, out_path: Path):
    corr = df_num.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0)
    plt.title('Matriz de correlación (features numéricos)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_like(y_true, y_pred, out_path: Path, n_bins=4):
    # Construir bins basados en cuantiles del y_true y asignar predicciones a esos bins.
    # Esto evita NaNs cuando y_pred es constante.
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    # calcular bordes por cuantiles del verdadero
    edges = np.quantile(y_true, q=np.linspace(0, 1, n_bins + 1))
    # asegurar monotonía y estabilidad (agregar epsilon a bordes repetidos)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6

    # asignar bins: etiquetas 0..n_bins-1
    q_true = np.digitize(y_true, edges[1:-1], right=True)
    # para predicciones, usar los mismos bordes
    q_pred = np.digitize(y_pred, edges[1:-1], right=True)

    cm = confusion_matrix(q_true, q_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted quantile bin')
    plt.ylabel('True quantile bin')
    plt.title(f'Confusion-like matrix (n_bins={n_bins})')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    print('Cargando datos...')
    df = pd.read_csv(DATA)

    print('Aplicando feature engineering...')
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    print('Preparando X/y (excluyendo u_casuales, u_registrados)...')
    X, y = prepare_X_y(df, target='total_alquileres', exclude=['u_casuales', 'u_registrados'])

    # Drop rows with missing y
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Matriz de correlación sobre X (numéricas)
    print('Generando matriz de correlación...')
    plot_corr_matrix(X, FIGS / 'correlation_matrix.png')
    print('Guardada:', FIGS / 'correlation_matrix.png')

    # Para matriz de confusión-like: necesitamos predicciones.
    # Usamos el modelo LGBM si existe, si no usamos catboost, sino predicción simple con media.
    model_path = ROOT / 'models' / 'model_lgbm.joblib'
    if not model_path.exists():
        # buscar otro modelo
        cand = list((ROOT / 'models').glob('model_*.joblib'))
        cand = [c for c in cand if 'quarantine' not in str(c)]
        model_path = cand[0] if cand else None

    if model_path is None:
        print('No se encontró modelo para predecir; usando media para predicción (fallback).')
        y_pred = np.repeat(y.median(), len(y))
    else:
        print('Cargando modelo:', model_path)
        model = joblib.load(model_path)
        # reindex X to model n_features if necessary
        try:
            X_pred = X.copy()
            # align columns if model has feature_name_
            if hasattr(model, 'feature_name_'):
                feat_names = list(model.feature_name_)
                # keep only features present in X
                feat_names = [f for f in feat_names if f in X_pred.columns]
                X_pred = X_pred.reindex(columns=feat_names, fill_value=0)
            y_pred = model.predict(X_pred)
        except Exception as e:
            print('Error predicting with model:', e)
            print('Usando media como fallback para predicción.')
            y_pred = np.repeat(y.median(), len(y))

    print('Generando matriz de confusión-like...')
    plot_confusion_like(y, y_pred, FIGS / 'confusion_like_matrix.png', n_bins=4)
    print('Guardada:', FIGS / 'confusion_like_matrix.png')

    # also save a small JSON summary
    summary = {
        'n_rows': int(len(df)),
        'n_features': int(X.shape[1]),
        'model_used': str(model_path)
    }
    with open(ROOT / 'reports' / 'figs' / 'matrices_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
