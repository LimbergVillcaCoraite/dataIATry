"""Script que calcula pares con alta correlación y sugiere features a eliminar
basado en importancias del `model_lgbm.joblib` (si está disponible).

Guarda recomendaciones en `reports/feature_selection_suggestion.json`.
"""
from pathlib import Path
import json

import joblib
import pandas as pd

from src.preprocessing import build_features, create_lag_features, prepare_X_y
from src.feature_selection import high_correlation_pairs, suggest_drops_by_importance


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset_alquiler.csv'
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)


def main():
    print('Cargando datos...')
    df = pd.read_csv(DATA)
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])

    print('Preparando X/y (excluyendo columnas prohibidas)...')
    X, y = prepare_X_y(df, target='total_alquileres', exclude=['u_casuales', 'u_registrados'])

    print('Calculando pares altamente correlacionados...')
    pairs = high_correlation_pairs(X, threshold=0.9)
    print(f'Encontrados {len(pairs)} pares con |corr|>=0.9')

    model_path = ROOT / 'models' / 'model_lgbm.joblib'
    model = None
    if model_path.exists():
        model = joblib.load(model_path)
        print('Cargado modelo LGBM para importancias')
    else:
        print('No se encontró model_lgbm.joblib; se usará proxy de varianza')

    drops = suggest_drops_by_importance(X, model, threshold=0.9)
    print('Sugerencia de drops:', drops)

    out = {
        'n_rows': int(len(X)),
        'n_features_before': int(X.shape[1]),
        'n_high_corr_pairs': len(pairs),
        'high_corr_pairs': [{'a': a, 'b': b, 'corr': c} for a, b, c in pairs],
        'suggested_drops': drops,
    }

    with open(OUT / 'feature_selection_suggestion.json', 'w') as f:
        json.dump(out, f, indent=2)

    print('Guardado en reports/feature_selection_suggestion.json')


if __name__ == '__main__':
    main()
