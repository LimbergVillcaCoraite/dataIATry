#!/usr/bin/env python3
"""Lista modelos en `models/` y exporta un resumen de métricas a `reports/models_metrics_summary.csv`.
"""
from pathlib import Path
import json
import csv


def main():
    repo = Path(__file__).resolve().parents[1]
    models_dir = repo / 'models'
    out_dir = repo / 'reports'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'models_metrics_summary.csv'

    rows = []
    for p in sorted(models_dir.glob('*.joblib')):
        name = p.stem
        metrics_file = p.with_suffix('.metrics.json')
        data = {'model': name, 'path': str(p), 'rmse_cv': '', 'mae_cv': '', 'rmse_in_sample': ''}
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as fh:
                    j = json.load(fh)
                data['rmse_cv'] = j.get('rmse_cv', '')
                data['mae_cv'] = j.get('mae_cv', '')
                data['rmse_in_sample'] = j.get('rmse_in_sample', '')
            except Exception as e:
                data['rmse_cv'] = f'error:{e}'
        else:
            data['rmse_cv'] = 'missing'
        rows.append(data)

    # escribir CSV
    with open(out_file, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['model', 'path', 'rmse_cv', 'mae_cv', 'rmse_in_sample'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote summary to: {out_file}")


if __name__ == '__main__':
    main()
