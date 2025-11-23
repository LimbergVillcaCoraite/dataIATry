import pandas as pd
import numpy as np
from pathlib import Path


def pct(x, total):
    return f"{(x/total*100):.2f}%"


def top_values(series, n=10):
    return series.value_counts(dropna=False).head(n)


def main():
    repo_root = Path(__file__).parent
    csv_path = repo_root / "dataset_alquiler.csv"
    report_path = repo_root / "analysis_report.md"

    if not csv_path.exists():
        print(f"ERROR: no existe {csv_path}")
        return

    print(f"Cargando {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    n_rows, n_cols = df.shape

    lines = []
    lines.append("# Informe de EDA - dataset_alquiler.csv")
    lines.append("")
    lines.append("## 1. Resumen general")
    lines.append(f"- Filas: {n_rows}")
    lines.append(f"- Columnas: {n_cols}")
    lines.append("")

    lines.append("## 2. Primeras filas")
    lines.append("```")
    lines.append(df.head(10).to_csv(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 3. Tipos de datos")
    types = df.dtypes.astype(str)
    for col, t in types.items():
        lines.append(f"- `{col}`: {t}")
    lines.append("")

    lines.append("## 4. Valores faltantes")
    missing = df.isna().sum()
    for col, miss in missing.items():
        if miss > 0:
            lines.append(f"- `{col}`: {miss} ({pct(miss, n_rows)})")
    if missing.sum() == 0:
        lines.append("- No se encontraron valores faltantes.")
    lines.append("")

    lines.append("## 5. Estadísticas descriptivas (numéricas)")
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        desc = num.describe().T
        # Evitar dependencia opcional 'tabulate' usando to_string
        lines.append(desc.to_string())
    else:
        lines.append("- No hay columnas numéricas detectadas.")
    lines.append("")

    lines.append("## 6. Estadísticas para variables categóricas")
    cat = df.select_dtypes(include=[object, 'category'])
    if not cat.empty:
        for col in cat.columns:
            lines.append(f"### `{col}`")
            vc = top_values(df[col], n=10)
            for val, cnt in vc.items():
                lines.append(f"- {repr(val)}: {cnt} ({pct(cnt, n_rows)})")
            lines.append("")
    else:
        lines.append("- No se detectaron variables categóricas (object/category).")
    lines.append("")

    lines.append("## 7. Correlaciones (Pearson) entre numéricas")
    if num.shape[1] >= 2:
        corr = num.corr()
        # Evitar dependencia opcional 'tabulate' usando to_string
        lines.append(corr.to_string())
        # Top correlation pairs (abs, exclude self)
        corr_unstack = corr.abs().unstack()
        corr_unstack = corr_unstack[corr_unstack < 1].sort_values(ascending=False).drop_duplicates()
        lines.append("")
        lines.append("### Pares con mayor correlación absoluta")
        for (a, b), val in corr_unstack.head(10).items():
            lines.append(f"- `{a}` & `{b}`: {val:.3f}")
    else:
        lines.append("- No hay suficientes columnas numéricas para calcular correlaciones.")
    lines.append("")

    lines.append("## 8. Duplicados y calidad")
    dup = df.duplicated().sum()
    lines.append(f"- Filas duplicadas exactas: {dup} ({pct(dup, n_rows)})")
    lines.append("")

    # Detectar columna fecha y parsear si existe
    if 'fecha' in df.columns:
        lines.append("## 9. Análisis de la columna `fecha`")
        try:
            fechas = pd.to_datetime(df['fecha'], errors='coerce')
            nan_dates = fechas.isna().sum()
            lines.append(f"- Fechas convertidas con {nan_dates} valores no convertibles.")
            lines.append(f"- Rango: {fechas.min()} — {fechas.max()}")
        except Exception as e:
            lines.append(f"- No se pudo parsear `fecha`: {e}")
        lines.append("")

    lines.append("## 10. Recomendaciones rápidas")
    lines.append("- Revisar columnas con muchos NA y decidir imputación o eliminación.")
    lines.append("- Transformar variables de tiempo (`fecha`, `hora`, `mes`, `dia_semana`) a características útiles.")
    lines.append("- Revisar `total_alquileres` vs `u_casuales` + `u_registrados` para consistencia.")
    lines.append("- Escalar variables numéricas si se usan modelos sensibles a escala.")
    lines.append("- Evaluar correlaciones altas antes de usar modelos lineales, o aplicar selección de variables.")

    # Escribir el reporte
    report_text = '\n'.join(lines)
    report_path.write_text(report_text, encoding='utf-8')
    print(f"Reporte generado en {report_path}")


if __name__ == '__main__':
    main()
