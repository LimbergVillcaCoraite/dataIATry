"""Genera SHAP summary para `models/model_catboost_hpo.joblib`.
Guarda `reports/figs/shap_catboost_summary.png` y `reports/shap_catboost_top_features.json`.
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing import build_features, create_lag_features, prepare_X_y


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset_alquiler.csv'
FIGS = ROOT / 'reports' / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    print('Cargando datos...')
    df = pd.read_csv(DATA)
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])
    X, y = prepare_X_y(df, exclude=['u_casuales', 'u_registrados'])
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    model_path = ROOT / 'models' / 'model_catboost_hpo.joblib'
    if not model_path.exists():
        raise FileNotFoundError('No se encontró model_catboost_hpo.joblib en models/')

    print('Cargando modelo CatBoost desde', model_path)
    model = joblib.load(model_path)

    # Alinear X si model tiene feature_names_
    try:
        if hasattr(model, 'feature_names_'):
            feat_names = list(model.feature_names_)
            X = X.reindex(columns=feat_names, fill_value=0)
            print(f'Alineado X a {len(feat_names)} features según model.feature_names_')
        elif hasattr(model, 'feature_name_'):
            feat_names = list(model.feature_name_)
            X = X.reindex(columns=feat_names, fill_value=0)
            print(f'Alineado X a {len(feat_names)} features según model.feature_name_')
    except Exception as e:
        print('Advertencia: no se pudo alinear X con las features del modelo:', e)

    # intentar SHAP
    try:
        import shap

        print('Usando SHAP TreeExplainer para CatBoost (muestra para velocidad)')
        explainer = shap.TreeExplainer(model)
        sample_idx = np.random.default_rng(42).choice(len(X), size=min(2000, len(X)), replace=False)
        X_sample = X.iloc[sample_idx]
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(8, 10))
        shap.summary_plot(shap_values, X_sample, show=False)
        out = FIGS / 'shap_catboost_summary.png'
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feat_importance = sorted(zip(X_sample.columns.tolist(), mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True)
        top = [{'feature': f, 'mean_abs_shap': float(v)} for f, v in feat_importance[:30]]
        with open(ROOT / 'reports' / 'shap_catboost_top_features.json', 'w') as fh:
            json.dump({'model': str(model_path), 'top_features': top}, fh, indent=2)

        print('SHAP CatBoost guardado en', out)
        return

    except Exception as e:
        warnings.warn(f'SHAP no disponible o falló ({e}), usando permutation importance como fallback')

    # Fallback: permutation importance
    try:
        from sklearn.inspection import permutation_importance
        print('Usando permutation_importance (sample para velocidad)')
        sample_idx = np.random.default_rng(42).choice(len(X), size=min(2000, len(X)), replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        r = permutation_importance(model, X_sample, y_sample, n_repeats=10, random_state=42, n_jobs=1)
        importances = r.importances_mean
        feat_importance = sorted(zip(X_sample.columns.tolist(), importances.tolist()), key=lambda x: x[1], reverse=True)

        names = [f for f, _ in feat_importance[:30]]
        vals = [v for _, v in feat_importance[:30]]
        plt.figure(figsize=(8, 6))
        plt.barh(names[::-1], vals[::-1])
        plt.title('Permutation importance (mean) - CatBoost')
        plt.tight_layout()
        out = FIGS / 'perm_importance_catboost.png'
        plt.savefig(out, dpi=150)
        plt.close()

        top = [{'feature': f, 'perm_importance': float(v)} for f, v in feat_importance[:30]]
        with open(ROOT / 'reports' / 'shap_catboost_top_features.json', 'w') as fh:
            json.dump({'model': str(model_path), 'top_features': top, 'note': 'permutation_importance fallback'}, fh, indent=2)

        print('Permutation importance CatBoost guardado en', out)
        return
    except Exception as e:
        raise RuntimeError('No se pudo generar SHAP ni permutation importance para CatBoost: ' + str(e))


if __name__ == '__main__':
    main()
