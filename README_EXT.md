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
