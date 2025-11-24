"""Retrain the best model from `reports/comparison_table.csv` on the full dataset.
Saves `models/model_<best>_final.joblib` and metrics JSON.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.preprocessing import prepare_X_y


def load_comparison(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    # drop rows with missing rmse
    df = df[df["rmse"].notna()]
    df = df.sort_values("rmse")
    if df.empty:
        raise RuntimeError("No valid models found in comparison table")
    return df.iloc[0]["model"]


def model_from_name(name: str):
    # names in comparison_table are like 'rf', 'xgb', etc.
    if name.startswith("model_"):
        name_clean = name.replace("model_", "")
    else:
        name_clean = name

    # Try to load candidate joblib if present
    candidate = Path("models") / f"model_{name_clean}_candidate.joblib"
    if candidate.exists():
        try:
            return joblib.load(candidate), name_clean
        except Exception:
            pass

    # Fallback: instantiate a default estimator for common names
    if name_clean in ("rf", "randomforest"):
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), name_clean
    if name_clean in ("linear",):
        return LinearRegression(), name_clean

    # As a last resort, use RandomForest
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), name_clean


def main():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "reports" / "comparison_table.csv"
    best_model_row = load_comparison(csv_path)
    print("Best candidate from comparison table:", best_model_row)

    model_obj, model_key = model_from_name(best_model_row)

    df = pd.read_csv(root / "dataset_alquiler.csv")
    X, y = prepare_X_y(df)

    mask = X.isna().any(axis=1) | pd.isna(y)
    if mask.any():
        X = X.loc[~mask].copy()
        y = y.loc[~mask].copy()

    print(f"Training final model '{model_key}' on full dataset ({len(y)} samples)")
    model_obj.fit(X, y)

    out_model = root / "models" / f"model_{model_key}_final.joblib"
    out_metrics = root / "models" / f"model_{model_key}_final.metrics.json"
    joblib.dump(model_obj, out_model)

    preds = model_obj.predict(X)
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))

    metrics = {"model": f"model_{model_key}", "rmse_in_sample": rmse, "mae_in_sample": mae, "n_samples": int(len(y))}
    out_metrics.write_text(json.dumps(metrics, indent=2))
    print("Saved final model and metrics:", out_model, out_metrics)


if __name__ == "__main__":
    main()
