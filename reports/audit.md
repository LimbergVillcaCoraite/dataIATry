# Auditoría de integridad de modelos

Fecha: 2025-11-22

Resumen:

- Se detectó un modelo guardado como `model_lgbm_hpo.joblib` que incluye las columnas prohibidas `u_casuales` y `u_registrados` entre sus features. Esto provoca métricas irreales por fuga de datos.
- Para evitar uso accidental, el artefacto se ha movido a `models/quarantine/`.

Acciones realizadas:

1. Inspección del artefacto:
   - Tipo: `lightgbm.sklearn.LGBMRegressor`
   - Número de features detectadas: 20
   - Algunas de las features presentes: `u_casuales`, `u_registrados`, `hora`, `temperatura`, etc.

2. Movido a cuarentena:
   - `models/model_lgbm_hpo.joblib` → `models/quarantine/model_lgbm_hpo.joblib`
   - `models/model_lgbm_hpo.metrics.json` → `models/quarantine/model_lgbm_hpo.metrics.json`

3. Experimentos controlados ejecutados:
   - `scripts/quick_experiment.py`: RandomizedSearchCV con lags [1,24,48,168] y rolling [3,6,24,168], TimeSeriesSplit n_splits=5. Resultado: mean RMSE ≈ 62.37 (no mejora frente a RMSE ≈ 60.11).
   - `scripts/experiment_log1p.py`: HPO sobre target transformado con `log1p`, evaluado en escala original. Resultado: mean RMSE ≈ 67.04 (no mejora).

Recomendaciones:

- No usar `models/quarantine/model_lgbm_hpo.joblib` para evaluación ni despliegue.
- Revisar cualquier pipeline de HPO para garantizar que los lags se crean antes del split y emplear `TimeSeriesSplit` o rolling-origin para validación.
- Si se desea mejorar métricas, pruebas sugeridas:
  - Expandir HPO con más iteraciones o Bayesian optimization.
  - Probar stacking/ensembles.
  - Añadir features temporales e interacciones (hora×día, feriados enriquecidos).

Archivos relacionados:

- `README.md` (sección "Auditoría de modelos y nota de integridad").
- `scripts/quick_experiment.py` y `scripts/experiment_log1p.py` (scripts ejecutados).
- `models/quarantine/` (artefactos en cuarentena).

