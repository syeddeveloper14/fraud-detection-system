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
                "score": d.score,
                "model_action": d.model_action,
                "action": d.action,
                "rules": ",".join(h["name"] for h in d.rule_hits),
                "latency_ms": d.latency_ms,
            }
        )
    decisions = pd.DataFrame(records, index=stream.index)
    lat = decisions["latency_ms"].to_numpy()
    summary: dict[str, Any] = {
        "n": int(len(decisions)),
        "action_rates": decisions["action"].value_counts(normalize=True).to_dict(),
        "rule_overrides": int((decisions["action"] != decisions["model_action"]).sum()),
        "latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
        },
    }

    if offline_features is not None:
        online = pd.DataFrame(history_rows, index=stream.index)
        diffs = {}
        for col in HISTORY_FEATURES:
            a = online[col].to_numpy(float)
            b = offline_features.loc[stream.index, col].to_numpy(float)
            both_nan = np.isnan(a) & np.isnan(b)
            mismatch_nan = np.isnan(a) ^ np.isnan(b)
            delta = np.where(both_nan, 0.0, np.abs(a - b))
            diffs[col] = (
                float(np.inf)
                if mismatch_nan.any()
                else float(np.nanmax(delta) if len(delta) else 0.0)
            )
        summary["feature_parity_max_abs_diff"] = diffs
        summary["feature_parity_ok"] = all(v < 1e-6 for v in diffs.values())

    if LABEL in stream.columns and costs is not None:
        y = stream[LABEL].to_numpy()
        per = outcome_costs(y, stream["amt"].to_numpy(), costs)
        actions = decisions["action"].to_numpy()
        total = sum(float(per[a][actions == a].sum()) for a in ("approve", "review", "decline"))
        summary["cost_total"] = total
        summary["cost_approve_all"] = float(per["approve"].sum())
        summary["fraud_caught"] = float(
            ((y == 1) & (actions != "approve")).sum() / max((y == 1).sum(), 1)
        )
    return ReplayResult(decisions=decisions, summary=summary)
