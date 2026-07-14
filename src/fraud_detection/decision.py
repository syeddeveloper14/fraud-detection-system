"""Scoring + rules + reason codes -> one decision per transaction."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .features.online import OnlineFeatureStore
from .features.static import static_row
from .models.registry import ModelBundle
from .rules import SEVERITY, RuleEngine


@dataclass
class Decision:
    trans_num: str
    score: float
    action: str
    model_action: str
    reasons: list[dict[str, Any]] = field(default_factory=list)
    rule_hits: list[dict[str, str]] = field(default_factory=list)
    model_version: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionEngine:
    def __init__(
        self,
        bundle: ModelBundle,
        rules: RuleEngine | None = None,
        store: OnlineFeatureStore | None = None,
        explain: bool = True,
    ) -> None:
        self.bundle = bundle
        self.rules = rules or RuleEngine.empty()
        self.store = store or OnlineFeatureStore()
        self.explain = explain

    def features_for(self, txn: dict[str, Any]) -> dict[str, Any]:
        """Online features for one transaction (same names/values as the offline frame)."""
        row = static_row(txn)
        row.update(self.store.compute(txn))
        row["merchant"] = str(txn["merchant"])
        return row

    def decide(self, txn: dict[str, Any], update_state: bool = True) -> Decision:
        start = time.perf_counter()
        row = self.features_for(txn)
        model = self.bundle.model
        vector = model.row_vector(row)
        score = float(model.calibrated(model.predict_raw_matrix(vector))[0])
        model_action = self.bundle.thresholds.action(score)

        context = {**row, "cc_num": str(txn["cc_num"])}
        hits = self.rules.evaluate(context)
        action = max([model_action, *(h.action for h in hits)], key=SEVERITY.__getitem__)

        reasons: list[dict[str, Any]] = []
        if self.explain and model_action != "approve":
            frame = pd.DataFrame(vector, columns=model.spec.columns)
            reasons = self.bundle.reason_coder.explain(model, frame)
        if update_state:
            self.store.update(txn)
        return Decision(
            trans_num=str(txn["trans_num"]),
            score=score,
            action=action,
            model_action=model_action,
            reasons=reasons,
            rule_hits=[{"name": h.name, "action": h.action, "reason": h.reason} for h in hits],
            model_version=self.bundle.version,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
