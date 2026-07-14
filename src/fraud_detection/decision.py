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
