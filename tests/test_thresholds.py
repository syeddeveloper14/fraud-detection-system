import itertools

import numpy as np
import pytest

from fraud_detection.config import CostConfig
from fraud_detection.models import Thresholds, optimize_thresholds, total_cost
from fraud_detection.models.evaluation import precision_at_top, recall_at_fpr


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.05).astype(int)
    scores = np.clip(rng.normal(0.15 + 0.6 * y, 0.15), 0, 1).round(3)
    amt = rng.gamma(2, 80, n)
    return y, scores, amt


def test_actions():
    t = Thresholds(review=0.2, decline=0.8)
    assert [t.action(s) for s in (0.1, 0.2, 0.5, 0.8, 0.99)] == [
        "approve",
        "review",
        "review",
        "decline",
        "decline",
    ]
    assert list(t.actions(np.array([0.1, 0.5, 0.9]))) == ["approve", "review", "decline"]


