"""The online feature store must reproduce offline features exactly.

Training/serving skew is the most common silent failure in real-time ML systems. These tests
stream simulated data through the online path and compare every feature with the batch path.
"""

import numpy as np
import pandas as pd

from fraud_detection.features import (
    OnlineFeatureStore,
    history_features,
    static_features,
    static_row,
)
from fraud_detection.features.spec import HISTORY_FEATURES, STATIC_FEATURES
from fraud_detection.schema import SCORING_FIELDS


def _assert_close(a: np.ndarray, b: np.ndarray, name: str) -> None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert (np.isnan(a) == np.isnan(b)).all(), f"{name}: NaN pattern differs"
    mask = ~np.isnan(a)
    np.testing.assert_allclose(a[mask], b[mask], rtol=1e-9, atol=1e-6, err_msg=name)


def test_history_parity(small_df):
    offline = history_features(small_df)
    store = OnlineFeatureStore()
    online = pd.DataFrame(
        [store.compute_and_update(t) for t in small_df[SCORING_FIELDS].to_dict("records")],
        index=small_df.index,
    )
    for col in HISTORY_FEATURES:
        _assert_close(online[col], offline[col], col)


def test_static_parity(small_df):
    sample = small_df.sample(500, random_state=0)
    offline = static_features(sample)
