# Despliegue y uso

Pasos rápidos para reproducir el entorno, entrenar el modelo y ejecutar la API.

Requisitos:
- Python 3.10+ (en el contenedor se usó 3.12)
- Virtualenv o entorno equivalente

1) Crear y activar el virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Ejecutar tests

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

3) Entrenar el modelo CatBoost (genera `models/model_catboost_final.joblib`)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_catboost_full.py
```

4) Validar el modelo guardado

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_model_catboost.py
```

5) Ejecutar la API localmente

```bash
PYTHONPATH=. .venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8080
```

6) Notas sobre modelos grandes

- Evitamos subir artefactos binarios grandes al repo. Si necesitas distribuir modelos pesados, usa Git LFS o almacén externo (S3/Google Cloud Storage) y añade instrucciones de descarga en el README.
