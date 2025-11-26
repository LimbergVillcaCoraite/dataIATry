# dataIATry — Predicción de alquileres por hora (monopatines)

# dataIATry — Predicción de alquileres por hora (monopatines)

Resumen
-------
Proyecto para predecir `total_alquileres` por hora excluyendo las columnas `u_casuales` y `u_registrados` (requisito del reto).

Estado actual (artefactos principales)
- Notebook ejecutado: `reports/analysis_executed.ipynb`.
- Métricas globales (reports/metrics.json).
- Modelos: varios candidatos en `models/` (ver listado en `models/`).
- API de inferencia: `api/app.py` (FastAPI). Endpoints: `/`, `/models`, `/predict`.

Estructura del repositorio
- `src/` — preprocesado y utilidades (funciones reutilizables: `build_features`, `create_lag_features`, `prepare_X_y`, `get_feature_columns_example`).
- `notebooks/` — notebooks de análisis y experimentación.
- `models/` — `*.joblib` (modelos) y `*.metrics.json` (métricas por experimento).
- `reports/` — notebook ejecutado, métricas y figuras generadas.
- `api/` — servicio FastAPI con endpoints para inferencia.
- `tests/` — pruebas de integración para el endpoint `/predict`.

Quickstart (desarrollador)
1) Crear entorno e instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Levantar API (desarrollo):

```bash
uvicorn api.app:app --reload --port 8000

# luego probar:
curl -sS -X GET http://127.0.0.1:8000/ | jq
```

3) Petición de ejemplo a `/predict` (JSON):

```json
{ "data": {"fecha":"2019-01-01 12:00:00","temperatura":5.2,"clima":1}, "model": "model_ridge_candidate" }
```

Modelos (notas importantes)
- Por razones de tamaño, los artefactos de modelos grandes no se mantienen directamente en el historial Git principal. Se está usando Git LFS para gestionar `models/*.joblib`.
- Backup local creado durante limpieza: `../model_artifacts_backup/` (copia de los `.joblib` restaurada temporalmente).

Instrucciones sobre Git LFS y artefactos
- Este repositorio ahora usa Git LFS para los modelos (`models/*.joblib`). Para trabajar localmente con los modelos, asegúrate de tener Git LFS instalado:

```bash
# Instalar y habilitar Git LFS
git lfs install

# Después de clonar por primera vez, descarga los objetos LFS
git lfs pull
```

- Si por algún motivo necesitas los archivos de modelo directamente (backup): hay copias en `../model_artifacts_backup/` dentro del entorno de trabajo actual (si todavía existen). Es recomendable subir los artefactos grandes a un storage dedicado (S3 / Releases de GitHub) si quieres compartirlos públicamente.

Problemas conocidos
- Si ves mensajes de "punteros" en lugar de archivos binarios al hacer `git show` o `cat` de un `.joblib`, asegúrate de ejecutar `git lfs pull` tras clonar.

Contacto
- Repo mantenido por `LimbergVillcaCoraite`.

Resumen
-------
Proyecto para predecir `total_alquileres` por hora excluyendo las columnas `u_casuales` y `u_registrados` (requisito del reto).

Estado actual (artefactos principales)
- Notebook ejecutado: `reports/analysis_executed.ipynb`.
- Métricas globales (reports/metrics.json):
  - RMSE: 3.2955
  - MAE: 1.0788
  - R2: 0.99966
- Modelos: varios candidatos en `models/` (ver listado más abajo).
- API de inferencia: `api/app.py` (FastAPI). Endpoints: `/`, `/models`, `/predict`.

Estructura del repositorio
- `src/` — preprocesado y utilidades (funciones reutilizables: `build_features`, `create_lag_features`, `prepare_X_y`, `get_feature_columns_example`).
- `notebooks/` — notebooks de análisis y experimentación.
- `models/` — `*.joblib` (modelos) y `*.metrics.json` (métricas por experimento).
- `reports/` — notebook ejecutado, métricas y figuras generadas.
- `api/` — servicio FastAPI con endpoints para inferencia.
- `tests/` — pruebas de integración para el endpoint `/predict`.

Quickstart (desarrollador)
1) Crear entorno e instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Levantar API (desarrollo):

```bash
uvicorn api.app:app --reload --port 8000

# luego probar:
curl -sS -X GET http://127.0.0.1:8000/ | jq
```

3) Petición de ejemplo a `/predict` (JSON):

```json
{ "data": {"fecha":"2019-01-01 12:00:00","temperatura":5.2,"clima":1}, "model": "model_ridge_candidate" }
```

Modelos (archivos presentes)
- `model_ada_candidate.metrics.json`
- `model_catboost_candidate.metrics.json`
- `model_catboost_final.metrics.json`
- `model_catboost_final_v1.metrics.json`
- `model_et_candidate.metrics.json`
- `model_gbr_candidate.metrics.json`
- `model_hgb_candidate.metrics.json`
- `model_knn_candidate.metrics.json`
- `model_lgbm_candidate.metrics.json`
- `model_linear_candidate.metrics.json`
- `model_rf_candidate.metrics.json`
- `model_ridge_candidate.joblib` (+ `model_ridge_candidate.metrics.json`)
- `model_xgb_candidate.metrics.json`

Preprocesado (detalles técnicos)
- Extracción de features temporales: `anio`, `mes`, `dia_semana`, `hora`, `es_feriado`, `es_dia_trabajo`.
- Lags: generación configurable (ej.: lag 1, 24, 168) en `src/preprocessing.py`.
- Exclusión explícita de columnas prohibidas: `u_casuales`, `u_registrados` — esto está implementado en `src/preprocessing.prepare_X_y`.
- Codificación: combinable entre one-hot y target/label encoding según experimento.
- Imputación: numéricos → mediana (por defecto); categóricos → categoría `missing`.

Evaluación y métricas
- Métrica principal: RMSE (penaliza errores grandes). Complemento: MAE (robusto) y R2 (ajuste global).
- `reports/metrics.json` contiene el resumen del experimento principal; archivos `models/*.metrics.json` contienen métricas por modelo y por folds cuando aplica.

Problemas conocidos y mitigaciones
- Feature-name mismatch para entradas unitarias en `/predict`: algunas cargas de modelos (scikit-learn) requieren que el DataFrame de entrada tenga exactamente las columnas usadas en fit. Se propone/implementa: reindexado de la fila de entrada con la lista de columnas de entrenamiento y rellenado con valores por defecto (0 o media histórica). La función `get_feature_columns_example` en `src/preprocessing.py` y el endpoint `/predict` en `api/app.py` contienen la lógica relacionada; el parche final se recomienda aplicar en `api/app.py`.
- Espacio en disco: modelos grandes se movieron a `artifacts/` cuando fue necesario; restaurados a `models/` para pruebas.

CI / Despliegue
- GitHub Actions: `.github/workflows/ci.yml` para tests y `.github/workflows/publish_image.yml` para construir/publish de imagen.
- GHCR image: `ghcr.io/limbergvillcacoraite/dataiatry:latest` (builds recientes OK).
- Render: `render.yaml` incluido; deploy automático requiere secrets (RENDER_API_KEY, RENDER_SERVICE_ID) y permisos para dispatch.

Objetivos del reto: estado y mapeo (¿se cumplen? dónde)
----------------------------------------------------
1) Entregar pipeline reproducible que prediga `total_alquileres` por hora y excluya `u_casuales`/`u_registrados`.
   - Estado: Cumplido (implementado).
   - Implementación / dónde: `src/preprocessing.py` (`prepare_X_y`) excluye esas columnas; notebooks en `notebooks/analysis.ipynb` muestran el pipeline y `reports/analysis_executed.ipynb` es la ejecución reproducible.

2) Incluir notebook reproducible con resultados y métricas ejecutadas.
   - Estado: Cumplido.
   - Implementación / dónde: `notebooks/analysis.ipynb` + `reports/analysis_executed.ipynb` y `reports/metrics.json`.

3) Proveer una API para inferencia (FastAPI) con endpoints documentados y tests básicos.
   - Estado: Parcialmente cumplido.
   - Implementación / dónde: `api/app.py` expone `/`, `/models`, `/predict`; tests en `tests/test_api_predict.py`. Pendiente robustecer `/predict` para entradas unitarias (reindex+imputación) — parche recomendado en `api/app.py`.

4) Añadir tests automatizados que verifiquen integraciones clave (API predict).
   - Estado: Parcial (tests presentes pero dependen de modelos en `models/` y del espacio/entorno para ejecutarse).
   - Implementación / dónde: `tests/test_api_predict.py`. Requiere que `models/*.joblib` exista en el entorno de CI/local.

5) Preparar CI/CD y artefactos para despliegue (Docker / GHCR / Render).
   - Estado: Cumplido (build/publish configurado); despliegue automático a Render requiere permisos adicionales para ejecutarse desde Actions.
   - Implementación / dónde: `.github/workflows/publish_image.yml`, `Dockerfile`, `render.yaml`.

6) Documentar claramente entregables y cómo reproducir el pipeline.
   - Estado: Cumplido (este README + `notebooks/` y `reports/`).
   - Implementación / dónde: `README.md`, `notebooks/analysis.ipynb`, `reports/analysis_executed.ipynb`.

Dónde buscar evidencias concretas
- Código de preprocesado y generación de features: `src/preprocessing.py`.
- Entrenamiento y comparativa de modelos: `notebooks/analysis.ipynb`, `scripts/` (si existen scripts de HPO/entrenamiento).
- Artefactos de modelos: `models/*.joblib` y `models/*.metrics.json`.
- Métricas agregadas: `reports/metrics.json`.
- API: `api/app.py` y tests `tests/test_api_predict.py`.
