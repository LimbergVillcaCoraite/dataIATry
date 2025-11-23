[![CI](https://github.com/LimbergVillcaCoraite/dataIATry/actions/workflows/ci.yml/badge.svg?branch=code-clean)](https://github.com/LimbergVillcaCoraite/dataIATry/actions/workflows/ci.yml)

# dataIATry

Resumen
-------

Proyecto de ejemplo para la prueba técnica: predicción del número de alquileres de monopatines por hora.

Estructura propuesta
- `src/`: código fuente (carga de datos, limpieza, entrenamiento)
- `api/`: servicio FastAPI para servir el modelo
- `models/`: modelos entrenados y métricas
- `notebooks/`: notebooks de análisis y visualizaciones
- `dataset_alquiler.csv`: dataset original
- `eda.py`: script de EDA rápido (ya incluido)

Requisitos
----------
Instala dependencias:

```bash
python3 -m pip install -r requirements.txt
```

# Data IATry — Predicción de alquileres de monopatines

Este repositorio contiene una solución completa (pipeline, modelos, API y artefactos) para el reto técnico: predecir el número de alquileres de monopatines por hora en Berlín.

He incluido pipelines reproducibles en Python, scripts de experimentación, y una API mínima para servir modelos entrenados. Todo el trabajo está pensado para ser reproducible localmente con el archivo `dataset_alquiler.csv`.

Contenido clave
 - `src/`: funciones de carga, preprocesamiento y entrenamiento.
 - `api/`: servicio FastAPI para predicciones en tiempo real y listado de modelos.
 - `models/`: modelos entrenados y archivos `*.metrics.json` con métricas resumidas.
 - `scripts/`: utilidades para HPO, SHAP, stacking y evaluación.
 - `reports/`: figuras y JSONs con resultados (SHAP, correlaciones, comparaciones).

Instalación rápida
1. Crear entorno y activar (recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ejecutar los scripts de ejemplo (desde la raíz):

```bash
# Buscar hiperparámetros para CatBoost (usa TimeSeriesSplit)
PYTHONPATH=. .venv/bin/python scripts/hpo_catboost.py

# Ejecutar Optuna para LGBM (100 trials)
PYTHONPATH=. .venv/bin/python scripts/optuna_lgbm.py --n_trials 100

# Generar SHAP para el CatBoost optimizado
PYTHONPATH=. .venv/bin/python scripts/run_shap_for_catboost.py

# Comparar modelos y guardar informe
PYTHONPATH=. .venv/bin/python scripts/evaluate_models.py
```

API (uso)

1. Levanta la API:

```bash
uvicorn api.app:app --reload --port 8000
```

2. Endpoints principales:
- `GET /` — información básica y modelos detectados.
- `GET /models` — lista de modelos disponibles y métricas (si existen).
- `POST /predict` — hace una predicción. Payload ejemplo:

```json
{
  "data": { "fecha": "2019-01-01 00:00:00", "hora": 0, "temperatura": 2.4, "clima": 1 },
  "model": "model_catboost_hpo"
}
```

El API descubre dinámicamente modelos en la carpeta `models/`. Por defecto la API intenta cargar `model_catboost_hpo` si existe.

Decisiones importantes y buenas prácticas
- Exclusión: los campos `u_casuales` y `u_registrados` se excluyen del entrenamiento y la API (eran columnas prohibidas para la prueba).
- Validación temporal: uso de `TimeSeriesSplit` en búsquedas de hiperparámetros para evitar fuga temporal.
- Explicabilidad: se generan gráficos SHAP para guiar selección de features; los top-features están en `reports/shap_catboost_top_features.json`.
- Quarantine: cualquier artefacto identificado con fuga de datos se movió a `models/quarantine/` y no debe usarse.

Pruebas
```bash
pytest -q
```

Limpieza y reproducibilidad
- Se eliminaron logs y archivos temporales de la rama de trabajo para mantener el repo limpio.
- `requirements.txt` contiene las dependencias necesarias; si deseas versiones fijas las podemos pinnear.

Siguientes pasos recomendados
- Revisar la selección final del modelo (por ejemplo `model_catboost_hpo`) y, si estás de acuerdo, re-entrenarlo sobre todo el dataset para un artefacto final.
- Empaquetar el servicio en Docker y desplegarlo (puedo generar `Dockerfile` y `docker-compose.yml` si lo deseas).

Si quieres, procedo a cualquiera de estas tareas: generar `Dockerfile`, reentrenar el modelo final y guardarlo con versión, o preparar el repo para despliegue en Render/Railway.

---

## CI / Integración continua

He incluido un workflow básico de GitHub Actions en `.github/workflows/ci.yml` que:
- instala dependencias en un virtualenv,
- ejecuta la suite de tests (`pytest`),
- (opcional) construye una imagen Docker si se activa el paso.

## Modelos grandes

No subimos modelos pesados al repositorio para evitar superar límites de GitHub. En su lugar:
- los artefactos `*.joblib` se deben publicar usando GitHub Releases o subir a un bucket S3/GCS; ver `RELEASE.md` y `scripts/publish_model.sh`.

## Despliegue rápido (Render)

He añadido una plantilla `render.yaml` con la configuración mínima para desplegar la API en Render. Para desplegar:

1. Conecta tu repo a Render y selecciona `web service`.
2. Usa el `Dockerfile` incluido o el servicio Python que expone `uvicorn api.app:app`.
3. Configura variables de entorno (si necesitas ruta a modelos en S3).

## Contacto y soporte

Si quieres, puedo:
- Añadir integración con Git LFS para almacenar modelos en el repo (requiere configuración del usuario).
- Añadir una GitHub Action para publicar modelos en Releases automáticamente.

Fin del README.

---

## Uso rápido de los nuevos scripts

He añadido dos scripts útiles que ejecuté en esta rama:

- `scripts/train_xgb_full.py`: reentrena XGBoost sobre todo el dataset usando los mejores hiperparámetros guardados por el HPO (`models/model_xgb_hpo.metrics.json`) si existen. Guarda el modelo en `models/model_xgb_final.joblib` y las métricas en `models/model_xgb_final.metrics.json`.

- `scripts/plot_comparison.py`: genera una figura comparativa de `RMSE` y `MAE` a partir de `reports/comparison_table.csv` y guarda la imagen en `reports/figs/comparison_rmse_mae.png`.

Comandos de ejemplo (desde la raíz del repo, con entorno activado):

```bash
# Ejecutar tests
PYTHONPATH=. ./.venv/bin/pytest -q

# Reentrenar XGBoost final
PYTHONPATH=. ./.venv/bin/python scripts/train_xgb_full.py

# Generar gráfica comparativa
./.venv/bin/python scripts/plot_comparison.py
```

La API por defecto ahora intenta cargar `model_catboost_final` si está disponible (esto facilita usar el modelo reentrenado final para inferencias). Si deseas que la API use otro modelo por defecto, edita `api/app.py`.

Si quieres, puedo crear un `Dockerfile` que incluya estos pasos para reproducir el entrenamiento y la API de forma aislada.

