"""HPO bayesiano con Optuna para LightGBM usando TimeSeriesSplit.

Guarda el mejor modelo en `models/model_lgbm_optuna.joblib` y métricas en
`models/model_lgbm_optuna.metrics.json`.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import optuna
import argparse

from src.preprocessing import build_features, create_lag_features, prepare_X_y

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset_alquiler.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)


def evaluate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def objective(trial, X, y, tscv):
    # parámetros a optimizar (se usarán en lgb.train)
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'verbose': -1,
    }

    n_estimators = trial.suggest_categorical('n_estimators', [100, 300, 500, 800])
    rmses = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val)
        bst = lgb.train(param, dtrain, num_boost_round=n_estimators, valid_sets=[dvalid], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)])
        preds = bst.predict(X_val, num_iteration=bst.best_iteration)
        rmses.append(evaluate_rmse(y_val, preds))

    return float(np.mean(rmses))


def run(n_trials: int = 50):
    print('Cargando datos...')
    df = pd.read_csv(DATA)
    df = build_features(df)
    df = create_lag_features(df, lags=[1, 24, 48, 168], rolling_windows=[3, 6, 24, 168])
    X, y = prepare_X_y(df, exclude=['u_casuales', 'u_registrados'])
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    tscv = TimeSeriesSplit(n_splits=5)

    study = optuna.create_study(direction='minimize')
    func = lambda trial: objective(trial, X, y, tscv)
    print(f'Iniciando Optuna (n_trials={n_trials})...')
    study.optimize(func, n_trials=n_trials)

    print('Mejor valor (RMSE):', study.best_value)
    print('Mejores params:', study.best_params)

    # Reentrenar mejor modelo sobre todo el X con params
    best_params = study.best_params
    # reconstruir parámetros para lgb.train
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': best_params.get('learning_rate'),
        'num_leaves': best_params.get('num_leaves'),
        'max_depth': best_params.get('max_depth'),
        'subsample': best_params.get('subsample'),
        'colsample_bytree': best_params.get('colsample_bytree'),
        'min_child_samples': best_params.get('min_child_samples'),
        'reg_alpha': best_params.get('reg_alpha'),
        'reg_lambda': best_params.get('reg_lambda'),
        'verbose': -1,
    }
    n_estimators = best_params.get('n_estimators', 500)
    dtrain = lgb.Dataset(X, label=y)
    bst = lgb.train(param, dtrain, num_boost_round=n_estimators)

    out_model = MODELS / 'model_lgbm_optuna.joblib'
    joblib.dump(bst, out_model)
    metrics = {'rmse': float(study.best_value), 'best_params': study.best_params}
    with open(out_model.with_suffix('.metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Modelo Optuna guardado en', out_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optuna HPO for LightGBM')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()
    run(n_trials=args.n_trials)
"""HPO bayesiano para LightGBM usando Optuna y TimeSeriesSplit.

Guarda el mejor modelo en `models/model_lgbm_optuna.joblib` y métricas en `.metrics.json` si mejora.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import lightgbm as lgb

from src.preprocessing import build_features, create_lag_features, prepare_X_y


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset_alquiler.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)


def objective(trial, X, y, tscv):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_categorical('n_estimators', [200, 400, 800]),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
    }

    rmses = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(**param, random_state=42, n_jobs=1)
        # Use callbacks for early stopping to be compatible with different LightGBM versions
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return float(np.mean(rmses))


def run(n_trials: int = 50):
    print('Cargando datos...')
    df = pd.read_csv(DATA)
    df = build_features(df)
    df = create_lag_features(df, lags=[1,24,48,168], rolling_windows=[3,6,24,168])
    X, y = prepare_X_y(df, exclude=['u_casuales', 'u_registrados'])
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    tscv = TimeSeriesSplit(n_splits=5)

    def obj(trial):
        return objective(trial, X, y, tscv)

    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=n_trials)

    print('Best trial:', study.best_trial.params)
    best_params = study.best_trial.params

    # Refit best model on full X
    best_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X, y)

    # Evaluate with TimeSeriesSplit to estimate generalization
    rmses = []
    maes = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        maes.append(mean_absolute_error(y_val, preds))

    mean_rmse = float(np.mean(rmses))
    mean_mae = float(np.mean(maes))
    metrics = {'rmse': mean_rmse, 'mae': mean_mae, 'best_params': best_params}

    # Compare with reference
    ref_path = MODELS / 'model_lgbm.metrics.json'
    ref_rmse = None
    if ref_path.exists():
        try:
            with open(ref_path, 'r') as f:
                ref = json.load(f)
                ref_rmse = float(ref.get('rmse'))
        except Exception:
            ref_rmse = None

    print('Optuna CV mean RMSE:', mean_rmse, 'ref RMSE:', ref_rmse)

    out_model = MODELS / 'model_lgbm_optuna.joblib'
    out_metrics = out_model.with_suffix('.metrics.json')
    if ref_rmse is None or mean_rmse < ref_rmse:
        joblib.dump(best_model, out_model)
        with open(out_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        print('Saved improved model to', out_model)
    else:
        print('No improvement over reference; model not saved')


if __name__ == '__main__':
    run(n_trials=50)
