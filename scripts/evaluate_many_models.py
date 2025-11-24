"""Train and evaluate multiple regression models using TimeSeriesSplit.
Saves per-model artifact (`models/model_<name>_candidate.joblib`) and metrics JSON.
"""
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from src.preprocessing import prepare_X_y


def load_hpo_params(path: Path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data.get("best_params", {}) or data.get("params", {})
    except Exception:
        return {}


def evaluate_model(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    maes = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        maes.append(mean_absolute_error(y_val, preds))
    return float(np.mean(rmses)), float(np.mean(maes)), rmses, maes


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "dataset_alquiler.csv")
    X, y = prepare_X_y(df)

    # drop NaN/Inf
    mask = X.isna().any(axis=1) | pd.isna(y)
    if mask.any():
        X = X.loc[~mask].copy()
        y = y.loc[~mask].copy()

    models = {}

    models["linear"] = LinearRegression()
    models["rf"] = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    models["et"] = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    models["gbr"] = GradientBoostingRegressor(n_estimators=200, random_state=42)
    models["hgb"] = HistGradientBoostingRegressor(max_iter=200, random_state=42)

    # optional external models with hpo params
    xgb_params = load_hpo_params(root / "models" / "model_xgb_hpo.metrics.json")
    if XGBRegressor is not None:
        try:
            params = {k: v for k, v in xgb_params.get("best_params", xgb_params).items() if k in XGBRegressor().__dict__ or True}
        except Exception:
            params = {}
        models["xgb"] = XGBRegressor(n_estimators=200, random_state=42, **params)

    lgbm_params = load_hpo_params(root / "models" / "model_lgbm_optuna.metrics.json")
    if LGBMRegressor is not None:
        models["lgbm"] = LGBMRegressor(n_estimators=200, random_state=42)

    cat_params = load_hpo_params(root / "models" / "model_catboost_hpo.metrics.json")
    if CatBoostRegressor is not None:
        models["catboost"] = CatBoostRegressor(verbose=0, random_state=42)

    out_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for name, model in models.items():
        print(f"Evaluating {name} ...")
        try:
            rmse, mae, rmses, maes = evaluate_model(model, X, y, n_splits=5)
        except Exception as e:
            print(f"Failed to evaluate {name}: {e}")
            continue

        # retrain on full data
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
        except Exception:
            pass

        model_file = out_dir / f"model_{name}_candidate.joblib"
        joblib.dump(model, model_file)

        metrics = {
            "model": f"model_{name}",
            "rmse_cv": rmse,
            "mae_cv": mae,
            "rmse_folds": rmses,
            "mae_folds": maes,
        }
        metrics_file = out_dir / f"model_{name}_candidate.metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2))

        summary.append({"model": name, "rmse": rmse, "mae": mae, "file": str(metrics_file)})

    # write comparison table
    import csv
    report = root / "reports"
    report.mkdir(parents=True, exist_ok=True)
    out_csv = report / "comparison_table.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "rmse", "mae", "file"])
        writer.writeheader()
        for row in sorted(summary, key=lambda x: (x["rmse"] is None, x["rmse"])):
            writer.writerow(row)

    print("Evaluation complete. Summary written to:", out_csv)


if __name__ == "__main__":
    main()
