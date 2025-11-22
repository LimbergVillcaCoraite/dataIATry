REPORT - Proyecto: Predicción de `total_alquileres`

Resumen ejecutivo
-----------------
Este repositorio contiene un pipeline reproducible para predecir `total_alquileres` sin utilizar las columnas prohibidas `u_casuales` y `u_registrados`.
Incluye: EDA, limpieza y feature engineering en `src/`, modelos guardados en `models/`, una API FastAPI en `api/app.py`, scripts para HPO y validación, y un notebook `notebooks/analysis.ipynb` que explica y demuestra los pasos.

Decisiones principales
---------------------
- Split temporal: se ordena por `fecha` y se realiza un split 80/20 para evitar fugas temporales.
- Lag features: `create_lag_features` crea lags y medias móviles sobre `total_alquileres` antes del split para que el conjunto de test tenga lags válidos sin leakage.
- Modelos: baseline RandomForest y LightGBM; HPO ligera para LightGBM (guardar mejores parámetros y métricas).
- API: `POST /predict` acepta payload con `{'data': {...}, 'model': '<model_stem>'}` y devuelve una predicción escalar.

Modelos y métricas (resumen)
----------------------------
Modelos disponibles en `models/` (archivo + métricas JSON):
- `model_baseline.joblib` (baseline)
- `model_rf.joblib` / `model_rf_tuned.joblib` - RandomForest (metrics en JSON)
- `model_lgbm.joblib`, `model_lgbm_tuned.joblib` - LightGBM
- `model_lgbm_hpo.joblib` - HPO (búsqueda previa)
- `model_lgbm_hpo_quick.joblib` - HPO rápida (guardada por el script `scripts/hpo_timeseries_quick.py`)

Ejemplos de métricas (ver `models/*.metrics.json` para detalles):
- `model_lgbm_tuned.metrics.json`: RMSE ≈ 67.77, MAE ≈ 45.32
- `model_lgbm_hpo_quick.metrics.json`: RMSE ≈ 9.28, MAE ≈ 2.98

Nota: la gran diferencia entre métricas sugiere investigar el origen — puede deberse a diferencias en particionado, columnas usadas o a un modelo que está sobreajustando con ciertos parámetros. Recomendación: comparar predicciones sobre el mismo test set y verificar columnas usadas en `prepare_X_y`.

Reproducibilidad - comandos clave
--------------------------------
Desde la raíz del repo:

- Instalar dependencias:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

- Ejecutar pruebas unitarias:

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

- Entrenar modelos (pipeline reproducible):

```bash
python -m src.train dataset_alquiler.csv models/
```

- Ejecutar HPO rápida (segura para desarrollo):

```bash
PYTHONPATH=. .venv/bin/python scripts/hpo_timeseries_quick.py
# o via Makefile
make hpo-quick
```

- Levantar API (local):

```bash
.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload
```

- Validar la entrega automáticamente:

```bash
python scripts/validate_submission.py
```

Sugerencias para la entrevista
------------------------------
- Explicar por qué se usan lags y medias móviles: capturan dependencia inmediata y estacionalidad horaria/diaria.
- Defender la elección de split temporal: evita fugas y refleja el uso en producción.
- Comentar sobre el HPO: mostramos búsqueda limitada por razones de tiempo/recursos; explicar trade-offs entre n_iter y costo computacional.
- Preparar respuestas ante preguntas sobre la discrepancia de métricas: mostrar cómo comparar usando el mismo test split y verificar columnas del modelo.

Próximos pasos recomendados
--------------------------
- Auditar columnas finales usadas por cada modelo (`X.columns`) y comparar contra `exclude` para asegurar cumplimiento del requerimiento.
- Ejecutar una HPO más controlada con Optuna y validación por TimeSeriesCV con embargo si es necesario.
- Añadir CI (GitHub Actions) que ejecute `pytest` y la validación automática al hacer PR.

Contacto
--------
Si quieres que prepare una versión final del informe en PDF o una presentación corta (2-3 slides) para la entrevista, puedo generarla a partir de este `REPORT.md`.
