"""Stream replay: push transactions one by one through the online path.

Used to (1) verify that online features match offline features exactly, (2) measure
end-to-end decision latency, and (3) estimate the business cost of a model + rules
configuration on a held-out period before deploying it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .decision import DecisionEngine
from .features.spec import HISTORY_FEATURES
from .models.registry import ModelBundle
from .models.thresholds import outcome_costs
from .rules import RuleEngine
from .schema import LABEL, SCORING_FIELDS


@dataclass
class ReplayResult:
    decisions: pd.DataFrame
    summary: dict[str, Any]


def replay(
    bundle: ModelBundle,
    history: pd.DataFrame,
    stream: pd.DataFrame,
    rules: RuleEngine | None = None,
    offline_features: pd.DataFrame | None = None,
    costs=None,
    explain: bool = False,
) -> ReplayResult:
    engine = DecisionEngine(bundle, rules=rules, explain=explain)
    engine.store.warm_up(history)

    records, history_rows = [], []
    for txn in stream[SCORING_FIELDS].to_dict("records"):
        history_rows.append(engine.store.compute(txn))
        d = engine.decide(txn)
        records.append(
            {
                "trans_num": d.trans_num,
