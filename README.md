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

Uso rápido
---------
Generar reporte EDA:

```bash
python3 eda.py
```

Entrenar modelo baseline:

```bash
python3 -m src.train
```

Levantar API (si ya entrenaste y existe `models/model_baseline.joblib`):

```bash
uvicorn api.app:app --reload --port 8000
```

Validación y resultados
-----------------------
- Ejecuta el validador para comprobar que la solución cumple los requisitos de la prueba técnica:

```bash
python3 scripts/validate_submission.py
```

- Métricas obtenidas (ejecución en el entorno):
	- `model_lgbm.joblib`: RMSE ≈ 60.11, MAE ≈ 38.82
	- `model_rf.joblib`: RMSE ≈ 69.81, MAE ≈ 45.13
	- Modelos adicionales y su métricas se encuentran en `models/*.metrics.json`.

Notebook
--------
- El notebook `notebooks/analysis.ipynb` ahora está formateado para ser visible en el entorno: contiene carga de datos, inspección rápida y muestra las figuras generadas en `reports/figs/`.

Cómo abrir el notebook
----------------------
- Desde la raíz del proyecto, abre Jupyter/Lab o usa VS Code: `code .` y abre `notebooks/analysis.ipynb`.\n
 - Para lanzar Jupyter desde el entorno virtual:

```bash
source .venv/bin/activate
jupyter lab
```

Buenas prácticas aplicadas
-------------------------
- Código modular en `src/` con `data.py`, `preprocessing.py`, `models.py` y `train.py`.
- Comentarios y documentación en el código explicando decisiones de preprocesamiento y modelado.
- Script de validación para verificar restricciones de la prueba técnica.

## Auditoría de modelos y nota de integridad

Durante la preparación de la entrega se detectó un artefacto de modelo (`model_lgbm_hpo.joblib`) que fue entrenado y guardado incluyendo las columnas prohibidas `u_casuales` y `u_registrados`. Ese modelo produce métricas irreales (RMSE ~ 8) debido a fuga de datos y **no** debe usarse en evaluación ni en producción. El artefacto se ha movido a `models/quarantine/`.

Las acciones tomadas:
- Se creó `models/quarantine/` y se movieron `model_lgbm_hpo.joblib` y su `.metrics.json` a esa carpeta.
- Se ejecutaron experimentos controlados (HPO rápido y reentrenamiento con transformación `log1p`) usando `TimeSeriesSplit` y creando lags antes de hacer los splits.
- Ninguno de los experimentos adicionales mejoró el modelo LGBM existente (RMSE de referencia: 60.11).

Recomendación: revisar cualquier artefacto antiguo antes de usarlo y preferir los modelos en `models/` que fueron entrenados siguiendo la política de exclusión de `u_casuales` y `u_registrados`.

Siguientes pasos (opcionales):
- Refinar búsqueda de hiperparámetros (RandomizedSearchCV / Bayesian) con Cross-Validation temporal.
- Desplegar API en Render/Railway y documentar el endpoint con ejemplos curl.

Ejecución en contenedor (opcional)
---------------------------------
Puedes crear un contenedor Docker para reproducir la API y el notebook. Aquí hay comandos rápidos para probar localmente:

```bash
# Construir imagen (desde la raíz)
docker build -t dataiatry:latest .

# Correr la API en un contenedor
docker run --rm -p 8000:8000 -v "$PWD":/app dataiatry:latest uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Si quieres que genere el `Dockerfile` y el `docker-compose.yml`, dime y los agrego.

Benchmark y mejoras implementadas
--------------------------------
Se ejecutó un benchmark simple (N=100) para medir latencia y tiempos de pipeline en este entorno:

- Pipeline (por petición): mean 0.0081s
- Cold load + predict (carga modelo por petición): mean 0.0059s
- Warm predict (modelo cacheado en memoria): mean 0.0008s

Mejoras aplicadas para rendimiento y limpieza de logs:
- Reducción de verbosidad de LightGBM (`verbosity=-1`).
- Conversión de categorías a `category` y uso de códigos ordinales en `encode_categoricals`.
- Cache de modelos en la API (`app.state.model_cache`) para reducir latencia en producción.
- Evitar creación de lags innecesarios para peticiones unitarias.
