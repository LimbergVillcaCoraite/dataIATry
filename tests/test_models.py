import pandas as pd
import numpy as np
from src.models import train_lightgbm


def test_train_lightgbm_predict():
    # tiny dataset
    X = pd.DataFrame({'f1': np.arange(20), 'f2': np.arange(20) * 2})
    y = X['f1'] * 0.5 + X['f2'] * 0.2 + 1.0
    model = train_lightgbm(X, y, n_estimators=10)
    preds = model.predict(X)
    assert len(preds) == len(y)
    # check reasonable rmse
    rmse = ((preds - y) ** 2).mean() ** 0.5
    assert rmse < 10
