"""Genera SHAP summary plot para `model_lgbm.joblib` y guarda top features.

Si `shap` no está disponible, cae en un fallback usando `permutation_importance`.
Guarda outputs en `reports/figs/shap_summary.png` y `reports/shap_top_features.json`.
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

    # Prefer LGBM variants (explicit order). Evitar modelos en quarantine.
    candidates = [ROOT / 'models' / 'model_lgbm.joblib', ROOT / 'models' / 'model_lgbm_tuned.joblib']
    model_path = None
    for c in candidates:
        if c.exists():
            model_path = c
            break
    if model_path is None:
        print('No se encontró LGBM preferente. Buscando cualquier modelo disponible...')
        cand = list((ROOT / 'models').glob('model_*.joblib'))
        cand = [c for c in cand if 'quarantine' not in str(c)]
        if not cand:
            raise FileNotFoundError('No se encontró ningún modelo en models/ para explicar')
        # prefer the one with feature_name_ matching X if possible
        chosen = None
        for c in cand:
            try:
                m = joblib.load(c)
                if hasattr(m, 'feature_name_') and len(m.feature_name_) == X.shape[1]:
                    chosen = c
                    break
            except Exception:
                continue
        model_path = chosen or cand[0]

    print('Cargando modelo desde', model_path)
    model = joblib.load(model_path)

    # Alinear X a las features que el modelo espera (si es posible)
    try:
        if hasattr(model, 'feature_name_'):
            feat_names = list(model.feature_name_)
            # mantener solo las columnas que el modelo conoce; rellenar faltantes con 0
            X = X.reindex(columns=feat_names, fill_value=0)
            print(f'Alineado X a {len(feat_names)} features según model.feature_name_')
        elif hasattr(model, 'n_features_in_'):
            n = int(model.n_features_in_)
            if n != X.shape[1]:
                # tomar primeras n columnas como heurística
                X = X.iloc[:, :n]
                print(f'Alineado X tomando las primeras {n} columnas para coincidir con n_features_in_')
    except Exception as e:
        print('Advertencia: no se pudo alinear X con las features del modelo:', e)

    # intentar SHAP
    try:
        import shap

        print('Usando SHAP TreeExplainer (muestra para velocidad)')
        # para modelos de árbol TreeExplainer es ideal
        explainer = shap.TreeExplainer(model)
        # sample para velocidad
        sample_idx = np.random.default_rng(42).choice(len(X), size=min(2000, len(X)), replace=False)
        X_sample = X.iloc[sample_idx]
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(8, 10))
        shap.summary_plot(shap_values, X_sample, show=False)
        out = FIGS / 'shap_summary.png'
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

        # calcular mean(|shap|) por feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feat_importance = sorted(zip(X_sample.columns.tolist(), mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True)
        top = [{'feature': f, 'mean_abs_shap': float(v)} for f, v in feat_importance[:30]]
        with open(ROOT / 'reports' / 'shap_top_features.json', 'w') as fh:
            json.dump({'model': str(model_path), 'top_features': top}, fh, indent=2)

        print('SHAP summary guardado en', out)
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

        # plot bar
        names = [f for f, _ in feat_importance[:30]]
        vals = [v for _, v in feat_importance[:30]]
        plt.figure(figsize=(8, 6))
        plt.barh(names[::-1], vals[::-1])
        plt.title('Permutation importance (mean)')
        plt.tight_layout()
        out = FIGS / 'perm_importance.png'
        plt.savefig(out, dpi=150)
        plt.close()

        top = [{'feature': f, 'perm_importance': float(v)} for f, v in feat_importance[:30]]
        with open(ROOT / 'reports' / 'shap_top_features.json', 'w') as fh:
            json.dump({'model': str(model_path), 'top_features': top, 'note': 'permutation_importance fallback'}, fh, indent=2)

        print('Permutation importance guardado en', out)
        return
    except Exception as e:
        raise RuntimeError('No se pudo generar SHAP ni permutation importance: ' + str(e))


if __name__ == '__main__':
    main()
