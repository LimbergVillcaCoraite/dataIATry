Este directorio contiene los modelos serializados usados por la API.

- `model_<name>_candidate.joblib`: modelos generados durante la evaluación (CV).
- `model_<name>_final.joblib`: modelo reentrenado sobre todo el dataset y destinado a producción.
- `*_hpo.joblib`: artefactos de HPO (pueden moverse a `artifacts/`).

Para publicar un modelo final use el siguiente patrón de versionado:

```
models/model_catboost_final_v1.joblib
models/model_catboost_final_v1.metrics.json
```

Para archivar HPO use: `python3 scripts/archive_artifacts.py`.
