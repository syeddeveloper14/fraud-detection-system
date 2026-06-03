import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import prepare_transactions, simulate_transactions, time_split
from fraud_detection.features import (
    FeatureSpec,
    TargetEncoder,
    haversine_km,
    history_features,
    static_features,
)
from fraud_detection.schema import RAW_COLUMNS


def _frame(rows):
    df = pd.DataFrame(
        rows, columns=["trans_num", "cc_num", "ts", "amt", "merchant", "merch_lat", "merch_long"]
    )
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def test_simulator_schema_and_determinism():
    a = simulate_transactions(n_cards=30, days=20, seed=3)
    b = simulate_transactions(n_cards=30, days=20, seed=3)
    assert list(a.columns) == RAW_COLUMNS
    assert a.equals(b)
    assert a["is_fraud"].isin([0, 1]).all()
    assert a["trans_num"].is_unique


def test_prepare_sorts_into_stream_order():
    df = prepare_transactions(
        simulate_transactions(n_cards=20, days=10, seed=1).sample(frac=1, random_state=0)
    )
    assert df["ts"].is_monotonic_increasing


def test_time_split_is_ordered(small_df):
    train, valid, test = time_split(small_df, 0.15, 0.2)
    assert train["ts"].max() <= valid["ts"].min()
    assert valid["ts"].max() <= test["ts"].min()
    assert len(train) + len(valid) + len(test) == len(small_df)
    with pytest.raises(ValueError):
        time_split(small_df, 0.6, 0.5)


def test_haversine_known_distance():
    # one degree of latitude ~ 111.2 km
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.1)
    assert haversine_km(10, 20, 10, 20) == 0


def test_static_features():
    df = pd.DataFrame(
        {
            "ts": ["2023-01-02 23:30:00"],
            "dob": ["1990-01-02"],
            "amt": [100.0],
            "lat": [0.0],
            "long": [0.0],
            "merch_lat": [1.0],
            "merch_long": [0.0],
            "city_pop": [999],
            "category": ["travel"],
        }
    )
    f = static_features(df).iloc[0]
    assert f["hour"] == 23 and f["is_night"] == 1 and f["day_of_week"] == 0
    assert f["age_years"] == pytest.approx(33.0, abs=0.01)
    assert f["distance_home_km"] == pytest.approx(111.19, abs=0.1)
    assert f["log_amt"] == pytest.approx(np.log1p(100))
