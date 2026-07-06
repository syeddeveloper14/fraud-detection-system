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


def test_optimizer_matches_brute_force(scored):
    y, scores, amt = scored
    costs = CostConfig(max_review_rate=None)
    best, cost = optimize_thresholds(y, amt, scores, costs, n_candidates=10_000)
    candidates = np.r_[np.unique(scores), 2.0]
    brute = min(
        total_cost(y, amt, scores, Thresholds(r, d), costs)["total"]
        for r, d in itertools.product(candidates, candidates)
        if r <= d
    )
    assert cost["total"] == pytest.approx(brute)
    assert best.review <= best.decline


def test_review_capacity_respected(scored):
    y, scores, amt = scored
    costs = CostConfig(review_cost=0.01, decline_fp_cost=1000.0, max_review_rate=0.02)
    best, _ = optimize_thresholds(y, amt, scores, costs)
    assert (best.actions(scores) == "review").mean() <= 0.02 + 1e-9


def test_cost_breakdown_adds_up(scored):
    y, scores, amt = scored
    c = total_cost(y, amt, scores, Thresholds(0.3, 0.7), CostConfig())
    assert c["total"] == pytest.approx(c["approve"] + c["review"] + c["decline"])


def test_ranking_metrics():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert recall_at_fpr(y, s, 0.0) == 1.0
    assert precision_at_top(y, s, 0.4) == 1.0
