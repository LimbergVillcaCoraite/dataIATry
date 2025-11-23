import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.app as app_module


client = TestClient(app_module.app)


def load_sample():
    root = Path(__file__).resolve().parents[1]
    dfp = root / 'dataset_alquiler.csv'
    import pandas as pd
    df = pd.read_csv(dfp)
    # take first row and convert to dict dropping target
    row = df.iloc[0].to_dict()
    for k in ['total_alquileres', 'u_casuales', 'u_registrados']:
        row.pop(k, None)
    return row


def test_root():
    r = client.get('/')
    assert r.status_code == 200
    data = r.json()
    assert 'service' in data


def test_models_list():
    r = client.get('/models')
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_predict_sample():
    sample = load_sample()
    payload = {'data': sample, 'model': 'model_catboost_hpo'}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    assert 'prediction' in j
    assert isinstance(j['prediction'], (int, float))
