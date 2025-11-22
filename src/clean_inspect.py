from pathlib import Path
import pandas as pd
import numpy as np
from src.data import load_data, basic_clean


def inspect_weather(csv_path: str | Path, out_md: str | Path):
    df = load_data(csv_path)
    report_lines = []
    for col in ['sensacion_termica', 'velocidad_viento']:
        report_lines.append(f"## Columna: {col}")
        if col not in df.columns:
            report_lines.append("- No existe la columna")
            continue
        vals = df[col].dropna().astype(float)
        report_lines.append(f"- count: {len(vals)}")
        report_lines.append(f"- min: {vals.min()}")
        report_lines.append(f"- median: {vals.median()}")
        report_lines.append(f"- max: {vals.max()}")
        report_lines.append("")

    # aplicar limpieza básica
    df2 = basic_clean(df.copy())
    report_lines.append("# Después de basic_clean()")
    for col in ['sensacion_termica', 'velocidad_viento']:
        report_lines.append(f"## Columna: {col}")
        if col not in df2.columns:
            report_lines.append("- No existe la columna")
            continue
        vals = df2[col].dropna().astype(float)
        report_lines.append(f"- count: {len(vals)}")
        report_lines.append(f"- min: {vals.min()}")
        report_lines.append(f"- median: {vals.median()}")
        report_lines.append(f"- max: {vals.max()}")
        report_lines.append("")

    Path(out_md).write_text('\n'.join(report_lines), encoding='utf-8')
    print(f"Reporte de limpieza generado en {out_md}")


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[1]
    csv = repo / 'dataset_alquiler.csv'
    out = repo / 'models' / 'cleaning_report.md'
    inspect_weather(csv, out)
