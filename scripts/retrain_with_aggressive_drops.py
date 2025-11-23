"""Aplica los drops sugeridos en reports/feature_selection_suggestion.json y reentrena.
Guarda modelos en `models/` y métricas en `reports/aggressive_drop_result.json`.
"""
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUGGEST = ROOT / 'reports' / 'feature_selection_suggestion.json'
SRC = ROOT / 'dataset_alquiler.csv'
TMP = ROOT / 'dataset_alquiler_aggressive.csv'
OUT = ROOT / 'reports' / 'aggressive_drop_result.json'


def make_aggressive_csv():
    print('Cargando sugerencias de drops...')
    if not SUGGEST.exists():
        raise FileNotFoundError('No se encontró reports/feature_selection_suggestion.json')
    with open(SUGGEST, 'r') as f:
        data = json.load(f)
    drops = data.get('suggested_drops', [])

    print('Leyendo dataset original...')
    df = pd.read_csv(SRC, low_memory=False)
    to_drop = [c for c in drops if c in df.columns]
    print('Columnas a eliminar (existentes):', to_drop)
    df = df.drop(columns=to_drop)
    df.to_csv(TMP, index=False)
    return TMP, to_drop


def run_train(tmp_csv, drops):
    import src.train as trainer
    out_dir = ROOT / 'models'
    trainer.train_all(tmp_csv, out_dir)

    # summarize metrics for models produced
    metrics = {}
    for p in out_dir.glob('*.metrics.json'):
        try:
            with open(p, 'r') as f:
                metrics[p.stem] = json.load(f)
        except Exception:
            continue

    result = {'dropped_columns': drops, 'metrics': metrics}
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2)
    print('Resultados guardados en', OUT)


if __name__ == '__main__':
    tmp, drops = make_aggressive_csv()
    run_train(tmp, drops)
