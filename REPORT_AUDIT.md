# Auditoría rápida vs. requisitos del desafío

Resumen del estado actual del repositorio y mapeo con los objetivos del reto.

1. Construcción del modelo
- Estado: completado. Hay scripts de entrenamiento y modelos guardados (`scripts/`, `models/`).
- Nota: modelos grandes no se versionan en git; `models/model_catboost_final.joblib` existe localmente en el workspace pero la política de repo evita subir binarios muy grandes.

2. Pipeline de ML
- Estado: completado. `src/preprocessing.py`, `src/models.py` y scripts en `scripts/` implementan transformaciones, lags, y entrenamiento reproducible.

3. API para predicciones
- Estado: completado. `api/app.py` expone `GET /`, `GET /models` y `POST /predict`. Soporta selección de modelo por nombre y manejo de LightGBM/CatBoost.

4. Funcionalidad extensible
- Estado: implementada. El endpoint `/models` lista artefactos y es posible pasar `model` en `POST /predict`.

5. Repositorio y reproducibilidad
- Estado: completado parcialmente. `DEPLOY.md` y `README.md` con instrucciones básicas están presentes; añadí CI (workflow) y documentación para publicar modelos grandes fuera de git.

6. Despliegue en plataforma gratuita
- Estado: pendiente. Añadí una plantilla `render.yaml` y pasos en `DEPLOY.md`. Requiere que el usuario conecte el repo a Render o Railway.

Conclusión y recomendaciones
- Reentrenar el modelo final (`model_catboost_final.joblib`) sobre todo el dataset (ya existe script `scripts/train_catboost_full.py`).
- Publicar modelos grandes en GitHub Releases o S3; añadí `RELEASE.md` y `scripts/publish_model.sh` para facilitarlo.
- Revisar `README.md` y añadir credenciales/secretos si se desea despliegue automatizado.
