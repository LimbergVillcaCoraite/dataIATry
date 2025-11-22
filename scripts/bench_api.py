"""Benchmark simple de latencia de inferencia (cold vs warm) y tiempo de pipeline.

Ejecutar desde la raíz del repo con: PYTHONPATH=. python scripts/bench_api.py
"""
import time
from pathlib import Path
import joblib
import pandas as pd
from src.preprocessing import build_features, create_lag_features, encode_categoricals, get_feature_columns_example

ROOT = Path('.').resolve()
DATA_PATH = ROOT / 'dataset_alquiler.csv'
MODEL_NAME = 'model_lgbm_tuned'
MODEL_PATH = ROOT / 'models' / f'{MODEL_NAME}.joblib'

N = 100

if not DATA_PATH.exists():
    raise SystemExit(f"Dataset not found: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found: {MODEL_PATH}")

# preparar muestra
_df = pd.read_csv(DATA_PATH, low_memory=False)
_df = build_features(_df)
_df = create_lag_features(_df)

# expected columns must be derived from a df that contains the target
expected_cols = get_feature_columns_example(_df)

sample = _df.drop(columns=['total_alquileres'], errors='ignore').tail(1)
if sample.shape[0] == 0:
    raise SystemExit('No sample row available for benchmark')

sample_dict = sample.to_dict(orient='records')[0]

# función pipeline + align
# expected_cols was computed above from a full df with target

def run_pipeline_once(data_dict):
    df = pd.DataFrame([data_dict])
    df = build_features(df)
    # no tiene sentido crear lags en una sola fila (no hay histórico), pero medimos coste
    df = create_lag_features(df)
    df = encode_categoricals(df)
    X = df.select_dtypes(include=['number']).reindex(columns=expected_cols, fill_value=0).fillna(0)
    return X

print('Benchmark: N =', N)
# medir pipeline time per request
ptimes = []
for i in range(N):
    t0 = time.perf_counter()
    X = run_pipeline_once(sample_dict)
    t1 = time.perf_counter()
    ptimes.append(t1 - t0)

# Load model once to get feature order for alignment (but we'll reload during cold test)
model0 = joblib.load(MODEL_PATH)
model_feature_names = None
if hasattr(model0, 'feature_name_'):
    model_feature_names = model0.feature_name_
elif hasattr(model0, 'booster_') and hasattr(model0.booster_, 'feature_name'):
    try:
        model_feature_names = model0.booster_.feature_name()
    except Exception:
        model_feature_names = None

# align a single X to the model feature order
X = run_pipeline_once(sample_dict)
if model_feature_names is not None:
    X = X.reindex(columns=model_feature_names, fill_value=0).fillna(0)
else:
    if hasattr(model0, 'n_features_in_'):
        n_expected = int(model0.n_features_in_)
        cols = list(X.columns)[:n_expected]
        X = X.reindex(columns=cols, fill_value=0).fillna(0)

# cold load: load model each time (model reloaded per request)
cold_times = []
for i in range(N):
    t0 = time.perf_counter()
    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)
    t1 = time.perf_counter()
    cold_times.append(t1 - t0)

# warm: load once
model = joblib.load(MODEL_PATH)
warm_times = []
# align columns to model's feature order if possible
model_feature_names = None
if hasattr(model, 'feature_name_'):
    model_feature_names = model.feature_name_
elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
    try:
        model_feature_names = model.booster_.feature_name()
    except Exception:
        model_feature_names = None

if model_feature_names is not None:
    # reindex X to model's feature order
    X = X.reindex(columns=model_feature_names, fill_value=0).fillna(0)
else:
    # fallback: if n_features_in_ available, truncate/expand expected_cols
    if hasattr(model, 'n_features_in_'):
        n_expected = int(model.n_features_in_)
        cols = list(X.columns)[:n_expected]
        X = X.reindex(columns=cols, fill_value=0).fillna(0)

for i in range(N):
    t0 = time.perf_counter()
    preds = model.predict(X)
    t1 = time.perf_counter()
    warm_times.append(t1 - t0)

import statistics
print('\nPipeline time (per request):')
print('  mean: {:.4f}s, median: {:.4f}s, p95: {:.4f}s'.format(statistics.mean(ptimes), statistics.median(ptimes), max(ptimes)))
print('\nCold load + predict (load model each request):')
print('  mean: {:.4f}s, median: {:.4f}s, p95: {:.4f}s'.format(statistics.mean(cold_times), statistics.median(cold_times), max(cold_times)))
print('\nWarm predict (model cached):')
print('  mean: {:.4f}s, median: {:.4f}s, p95: {:.4f}s'.format(statistics.mean(warm_times), statistics.median(warm_times), max(warm_times)))

print('\nSummary: total elapsed')
print('  total pipeline:', sum(ptimes))
print('  total cold:', sum(cold_times))
print('  total warm:', sum(warm_times))
