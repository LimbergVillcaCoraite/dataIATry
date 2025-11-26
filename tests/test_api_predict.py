from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


def test_predict_catboost_final():
    payload = {
        "data": {
            "fecha": "2023-08-01 12:00:00",
            "total_alquileres": 100,
            "u_casuales": 10,
            "u_registrados": 90
        },
        "model": "model_catboost_final"
    }
    r = client.post('/predict', json=payload)
    assert r.status_code == 200, f"status {r.status_code} body={r.text}"
    j = r.json()
    assert 'prediction' in j
    assert isinstance(j['prediction'], (int, float))
