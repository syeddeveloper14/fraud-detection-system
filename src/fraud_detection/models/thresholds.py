"""Cost-based decision thresholds.

Every transaction ends in one of three actions. Costs per outcome:

| truth \\ action | approve                 | review                                     | decline        |
|-----------------|-------------------------|--------------------------------------------|----------------|
| fraud           | amount + chargeback fee | review cost + (1 - catch rate) x (amount + fee) | 0         |
| legitimate      | 0                       | review cost                                | friction cost  |

Two thresholds split the score range: decline if p >= t_decline, review if
t_review <= p < t_decline, approve otherwise. They are chosen jointly to minimise total cost,
optionally under a cap on the share of traffic sent to manual review (analyst capacity).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..config import CostConfig

ACTIONS = ("approve", "review", "decline")


@dataclass
class Thresholds:
    review: float
    decline: float

    def action(self, score: float) -> str:
        if score >= self.decline:
            return "decline"
        if score >= self.review:
            return "review"
        return "approve"

    def actions(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)
        return np.where(
            scores >= self.decline, "decline", np.where(scores >= self.review, "review", "approve")
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def outcome_costs(y: np.ndarray, amt: np.ndarray, costs: CostConfig) -> dict[str, np.ndarray]:
    y = np.asarray(y).astype(bool)
    amt = np.asarray(amt, dtype=float)
    loss = amt + costs.chargeback_fee
    return {
        "approve": np.where(y, loss, 0.0),
        "review": costs.review_cost + np.where(y, (1 - costs.review_catch_rate) * loss, 0.0),
        "decline": np.where(y, 0.0, costs.decline_fp_cost),
    }


def total_cost(
    y: np.ndarray, amt: np.ndarray, scores: np.ndarray, thresholds: Thresholds, costs: CostConfig
) -> dict[str, float]:
    per = outcome_costs(y, amt, costs)
    actions = thresholds.actions(scores)
    breakdown = {a: float(per[a][actions == a].sum()) for a in ACTIONS}
    breakdown["total"] = float(sum(breakdown.values()))
    return breakdown


def optimize_thresholds(
    y: np.ndarray,
    amt: np.ndarray,
    scores: np.ndarray,
    costs: CostConfig,
    n_candidates: int = 400,
) -> tuple[Thresholds, dict[str, float]]:
    """Grid search over (review, decline) cut points in O(n log n + candidates^2).

    Rows are sorted by score; for cut positions k_d <= k_r the top k_d rows are declined,
    the next k_r - k_d reviewed and the rest approved. Prefix sums give each pair's cost in
    O(1), so the whole grid is evaluated with array broadcasting.
    """
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    per = outcome_costs(np.asarray(y)[order], np.asarray(amt)[order], costs)
    zero = np.zeros(1)
    cum = {a: np.concatenate([zero, np.cumsum(per[a])]) for a in ACTIONS}
    n = len(s)

    # Candidate cut positions: only where the score changes (ties must stay together).
    boundaries = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])  # first index of each score value
    cuts = np.unique(np.r_[0, boundaries, n])
    if len(cuts) > n_candidates:
        cuts = np.unique(np.r_[0, cuts[np.linspace(0, len(cuts) - 1, n_candidates).astype(int)], n])

    kd = cuts[:, None]
    kr = cuts[None, :]
    cost = (
        cum["decline"][kd]
        + (cum["review"][kr] - cum["review"][kd])
        + (cum["approve"][n] - cum["approve"][kr])
    )
    valid = kr >= kd
    if costs.max_review_rate is not None:
        valid &= (kr - kd) <= costs.max_review_rate * n
    cost = np.where(valid, cost, np.inf)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    k_d, k_r = int(cuts[i]), int(cuts[j])

    def threshold_at(k: int) -> float:
        # Smallest score inside the top-k block; above every score when k == 0.
        return float(s[k - 1]) if k > 0 else float(np.nextafter(s[0], np.inf)) if n else 1.0

    decline_t = threshold_at(k_d)
    review_t = min(threshold_at(k_r), decline_t)
    thresholds = Thresholds(review=review_t, decline=decline_t)
    return thresholds, total_cost(y, amt, scores, thresholds, costs)
