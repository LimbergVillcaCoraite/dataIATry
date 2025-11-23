# Publicar modelos grandes

No se recomienda versionar artefactos binarios grandes directamente en git. Opciones recomendadas:

1) GitHub Releases
- Empaqueta `models/model_catboost_final.joblib` y sube como release asset.
- Añade notas con métricas (ej. contenido de `models/model_catboost_final.metrics.json`).

2) Almacenamiento en la nube (S3/GCS)
- Subir el artefacto al bucket y compartir URL pre-firmada en README.

Script de ayuda
- `scripts/publish_model.sh` empaqueta y sugiere comandos `gh` para crear una release.
