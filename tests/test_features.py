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


def test_window_features_exclude_current_and_simultaneous_events():
    df = _frame(
        [
            ("a", "c1", "2023-01-01 10:00:00", 10.0, "m1", 40.0, -90.0),
            ("b", "c1", "2023-01-01 10:30:00", 20.0, "m2", 40.0, -90.0),
            ("c", "c1", "2023-01-01 10:30:00", 30.0, "m1", 40.0, -90.0),  # same second as b
            ("d", "c1", "2023-01-01 11:15:00", 40.0, "m3", 41.0, -90.0),
            ("e", "c2", "2023-01-01 11:16:00", 50.0, "m1", 40.0, -90.0),
        ]
    )
    h = history_features(df)
    assert h.loc[0, "txn_count_1h"] == 0 and h.loc[0, "is_new_merchant"] == 1
    assert h.loc[1, "txn_count_1h"] == 1
    # c happens at the same timestamp as b: b is not "before" c for window features ...
    assert h.loc[2, "txn_count_1h"] == 1 and h.loc[2, "amt_sum_1h"] == 10.0
    # ... but it is the previous event in stream order
    assert h.loc[2, "secs_since_last"] == 0
    assert h.loc[2, "is_new_merchant"] == 0  # m1 used by event a
    # d: the 1h window [10:15, 11:15) holds b and c, not a
    assert h.loc[3, "txn_count_1h"] == 2 and h.loc[3, "amt_sum_1h"] == 50.0
    assert h.loc[3, "txn_count_24h"] == 3
    assert h.loc[3, "km_from_last"] == pytest.approx(111.19, abs=0.1)
    assert h.loc[3, "speed_kmh"] == pytest.approx(111.19 / 0.75, rel=1e-3)
    assert h.loc[3, "n_prior_txns"] == 3
    # another card has its own history
    assert h.loc[4, "n_prior_txns"] == 0 and np.isnan(h.loc[4, "secs_since_last"])


def test_baseline_stats():
    df = _frame(
        [
            ("a", "c1", "2023-01-01", 10.0, "m", 0.0, 0.0),
            ("b", "c1", "2023-01-02", 20.0, "m", 0.0, 0.0),
            ("c", "c1", "2023-01-03", 60.0, "m", 0.0, 0.0),
        ]
    )
    h = history_features(df)
    assert np.isnan(h.loc[0, "amt_mean_30d"])
    assert np.isnan(h.loc[1, "amt_std_30d"])  # needs two prior points
    assert h.loc[2, "amt_mean_30d"] == pytest.approx(15.0)
    assert h.loc[2, "amt_std_30d"] == pytest.approx(np.std([10, 20], ddof=1))
    assert h.loc[2, "amt_ratio_30d"] == pytest.approx(60 / 16)


def test_target_encoder_smoothing_and_unknowns():
    keys = pd.Series(["a"] * 10 + ["b"] * 10)
    y = pd.Series([1] * 5 + [0] * 5 + [0] * 10)
    enc = TargetEncoder(smoothing=10).fit(keys, y)
    prior = 0.25
    assert enc.transform(["a"])[0] == pytest.approx((5 + 10 * prior) / 20)
    assert enc.transform(["zzz"])[0] == pytest.approx(prior)
    assert enc.transform_one("a") == pytest.approx(enc.transform(["a"])[0])


def test_target_encoder_oof_does_not_see_own_label():
    keys = pd.Series([f"k{i}" for i in range(50)])  # every key appears once
    y = pd.Series(np.r_[np.ones(25), np.zeros(25)])
    oof = TargetEncoder(smoothing=1, n_folds=5).fit_transform_oof(keys, y)
    # a unique key never appears in its own fitting fold, so it always gets a fold prior
    assert np.all((oof > 0.3) & (oof < 0.7))


def test_feature_spec_frame_and_row_agree():
    spec = FeatureSpec()
    values = {c: 1.0 for c in spec.numeric}
    values["category"] = "travel"
    frame = spec.to_model_frame(pd.DataFrame([values]))
    assert list(frame.columns) == spec.columns
    np.testing.assert_array_equal(frame.to_numpy(), spec.row_vector(values))
    values["category"] = "not-a-category"
    assert np.isnan(spec.row_vector(values)[0, -1])
    assert spec.categorical_mask.sum() == 1 and spec.categorical_mask[-1]
