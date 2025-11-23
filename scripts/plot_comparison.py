"""Generate a comparison bar chart (RMSE and MAE) from comparison CSV.
Saves image to `reports/figs/comparison_rmse_mae.png`.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "reports" / "comparison_table.csv"
    out_dir = root / "reports" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_rmse_mae.png"

    if not csv_path.exists():
        print("comparison_table.csv not found at", csv_path)
        return

    df = pd.read_csv(csv_path)

    # Expect columns: model, rmse, mae
    if "rmse" not in df.columns or "mae" not in df.columns:
        print("CSV missing expected columns 'rmse' and 'mae'. Found:", df.columns.tolist())
        return

    df_sorted = df.sort_values("rmse")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(df_sorted))))
    x = range(len(df_sorted))

    ax.barh(x, df_sorted["rmse"], height=0.4, label="RMSE", color="#1f77b4")
    ax.barh([i + 0.45 for i in x], df_sorted["mae"], height=0.4, label="MAE", color="#ff7f0e")

    ax.set_yticks([i + 0.2 for i in x])
    ax.set_yticklabels(df_sorted["model"])
    ax.set_xlabel("Error")
    ax.set_title("Comparación de métricas por modelo")
    ax.legend()
    plt.tight_layout()

    fig.savefig(out_path, dpi=150)
    print("Saved comparison plot to:", out_path)


if __name__ == "__main__":
    main()
