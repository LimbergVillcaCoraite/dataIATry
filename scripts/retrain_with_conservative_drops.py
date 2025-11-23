"""Genera un dataset temporal quitando duplicados obvios (anio, mes)
y llama al pipeline de entrenamiento `src.train.train_all`.
"""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'dataset_alquiler.csv'
TMP = ROOT / 'dataset_alquiler_reduced.csv'


def make_reduced_csv():
    print('Leyendo dataset original...')
    df = pd.read_csv(SRC, low_memory=False)
    to_drop = [c for c in ['anio', 'mes'] if c in df.columns]
    if to_drop:
        print('Eliminando columnas:', to_drop)
        df = df.drop(columns=to_drop)
    else:
        print('No se encontraron columnas anio/mes para eliminar')
    df.to_csv(TMP, index=False)
    print('CSV reducido guardado en', TMP)


def run_train():
    import src.train as trainer
    out = ROOT / 'models'
    trainer.train_all(TMP, out)


if __name__ == '__main__':
    make_reduced_csv()
    run_train()
