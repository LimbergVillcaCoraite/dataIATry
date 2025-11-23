#!/usr/bin/env bash
# Script auxiliar para empaquetar y subir un modelo a GitHub Releases usando gh (GitHub CLI)

set -euo pipefail

MODEL=${1:-models/model_catboost_final.joblib}
METRICS=${2:-models/model_catboost_final.metrics.json}
TAG=${3:-"model-$(date +%Y%m%d-%H%M)"}

if [ ! -f "$MODEL" ]; then
  echo "Modelo no encontrado: $MODEL"
  exit 1
fi

echo "Creando release $TAG y subiendo $MODEL"
if ! command -v gh >/dev/null 2>&1; then
  echo "Instala GitHub CLI (gh) para usar este script: https://cli.github.com/"
  exit 1
fi

gh release create "$TAG" "$MODEL" --title "$TAG" --notes-file "$METRICS" || {
  echo "Falló la creación de release con gh"
  exit 1
}

echo "Release creada: $TAG"
