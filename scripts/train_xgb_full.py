"""Train XGBoost on the full dataset using best params from HPO.
Saves model to `models/model_xgb_final.joblib` and metrics to
`models/model_xgb_final.metrics.json`.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

from src.preprocessing import prepare_X_y


def load_best_params(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    data = json.loads(metrics_path.read_text())
    return data.get("best_params", {}) or data.get("params", {})


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "dataset_alquiler.csv"
    metrics_hpo = root / "models" / "model_xgb_hpo.metrics.json"
    out_model = root / "models" / "model_xgb_final.joblib"
    out_metrics = root / "models" / "model_xgb_final.metrics.json"

    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)

    print("Preparing X, y... (this will drop forbidden columns)")
    X, y = prepare_X_y(df)

    # Drop rows with missing or infinite values in features or target
    mask_missing = X.isna().any(axis=1) | pd.isna(y)
    # also filter out infinite values
    mask_inf = ~np.isfinite(X.select_dtypes(include=["number"]).to_numpy()).all(axis=1)
    mask_target_inf = ~np.isfinite(y.to_numpy())
    mask = mask_missing | mask_inf | mask_target_inf
    if mask.any():
        n_drop = int(mask.sum())
        print(f"Dropping {n_drop} rows with NaN/Inf before training")
        X = X.loc[~mask].copy()
        y = y.loc[~mask].copy()

    best_params = load_best_params(metrics_hpo)
    if best_params:
        print("Loaded best params from HPO:", best_params)
    else:
        print("No HPO params found; using reasonable defaults for XGBoost.")
        best_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
        }

    # Ensure some params expected by scikit-learn API are present
    best_params.setdefault("random_state", 42)

    model = XGBRegressor(**best_params)

    print("Training XGBoost on full data...")
    model.fit(X, y)

    print("Saving model to:", out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)

    print("Computing in-sample metrics (RMSE / MAE)")
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))

    metrics = {
        "model": "model_xgb_final",
        "rmse_in_sample": rmse,
        "mae_in_sample": mae,
        "n_samples": int(len(y)),
        "best_params": best_params,
    }
    out_metrics.write_text(json.dumps(metrics, indent=2))
    print("Saved metrics to:", out_metrics)


if __name__ == "__main__":
    main()
