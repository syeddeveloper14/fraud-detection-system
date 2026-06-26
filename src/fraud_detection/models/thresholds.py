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


