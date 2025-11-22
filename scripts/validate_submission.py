"""Script de validación para comprobar requisitos de la prueba técnica.

Comprueba lo siguiente:
- Que `prepare_X_y` excluye `u_casuales` y `u_registrados` de la matriz de features.
- Que existen modelos guardados en `models/` y sus archivos de métricas.
- Resume hallazgos en `reports/validation_report.md`.

Este script es útil para verificar antes de subir el repositorio final.
"""
from pathlib import Path
import json
import sys
import pandas as pd

# Asegurar que el paquete `src` pueda importarse cuando se ejecute el script
repo = Path('.').resolve()
sys.path.insert(0, str(repo))
try:
    from src.preprocessing import build_features, prepare_X_y
except Exception as e:
    raise RuntimeError(f"No se pudo importar src.*: {e}")


def main():
    csv = repo / 'dataset_alquiler.csv'
    out = repo / 'reports'
    out.mkdir(exist_ok=True)

    df = pd.read_csv(csv, low_memory=False)
    df = build_features(df)

    report = []
    report.append('# Validation Report')

    # 1) prepare_X_y does not include u_casuales/u_registrados
    try:
        X, y = prepare_X_y(df.head(1000), exclude=['u_casuales', 'u_registrados'])
        cols = set(X.columns)
        violations = []
        for c in ['u_casuales', 'u_registrados']:
            if c in cols:
                violations.append(c)
        if violations:
            report.append('- ERROR: columnas prohibidas en X: ' + ', '.join(violations))
        else:
            report.append('- OK: `u_casuales` y `u_registrados` no están en X')
    except Exception as e:
        report.append(f'- ERROR ejecutando prepare_X_y: {e}')

    # 2) Check models directory
    models_dir = repo / 'models'
    model_files = list(models_dir.glob('*.joblib')) if models_dir.exists() else []
    if not model_files:
        report.append('- WARNING: no se encontraron modelos en `models/`')
    else:
        report.append(f'- Se encontraron {len(model_files)} modelo(s):')
        for m in model_files:
            report.append(f'  - {m.name}')
            metrics_file = m.with_suffix('.metrics.json')
            if metrics_file.exists():
                try:
                    metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
                    report.append(f'    - métricas: {metrics}')
                except Exception:
                    report.append('    - ERROR leyendo métricas')
            else:
                report.append('    - WARNING: no existe archivo de métricas (.metrics.json)')

    # Escribir reporte
    rpt_path = out / 'validation_report.md'
    rpt_path.write_text('\n'.join(report), encoding='utf-8')
    print('Reporte de validación generado en', rpt_path)


if __name__ == '__main__':
    main()
