import pandas as pd
from src.preprocessing import build_features, create_lag_features, prepare_X_y


def test_lags_and_prepare():
    # small sample
    data = {
        'fecha': pd.date_range('2021-01-01', periods=5, freq='H'),
        'hora': [0,1,2,3,4],
        'total_alquileres': [10, 12, 15, 13, 20],
        'u_casuales': [5,6,7,5,10],
        'u_registrados': [5,6,8,8,10]
    }
    df = pd.DataFrame(data)
    df = build_features(df)
    df = create_lag_features(df, lags=[1,2], rolling_windows=[2])

    # check lag columns
    assert 'lag_1' in df.columns
    assert 'lag_2' in df.columns
    assert 'rolling_mean_2' in df.columns

    # prepare X,y excluding user columns
    X, y = prepare_X_y(df, exclude=['u_casuales', 'u_registrados'])
    assert 'u_casuales' not in X.columns
    assert 'u_registrados' not in X.columns
    assert len(y) == len(df)
