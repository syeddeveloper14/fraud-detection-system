"""FastAPI scoring service.

Run with ``fraud-detect serve`` or ``uvicorn fraud_detection.api.app:create_app --factory``.
Configuration via environment variables:

* ``FRAUD_BUNDLE``  - bundle directory or artifacts root (default ``artifacts``)
* ``FRAUD_RULES``   - rules YAML (optional)
* ``FRAUD_WARMUP``  - CSV of recent transactions to pre-load card history (optional)
"""

from __future__ import annotations

import os
import threading
from collections import Counter, deque

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from ..data import load_transactions
from ..decision import DecisionEngine
from ..models.registry import ModelBundle, load_bundle
from ..rules import RuleEngine
from .schemas import HealthOut, ScoreOut, TransactionIn


class ServiceMetrics:
    def __init__(self, window: int = 10_000) -> None:
        self.actions: Counter[str] = Counter()
        self.errors = 0
        self.latencies: deque[float] = deque(maxlen=window)

    def observe(self, action: str, latency_ms: float) -> None:
        self.actions[action] += 1
        self.latencies.append(latency_ms)

    def prometheus(self, version: str) -> str:
        lines = [
            "# HELP fraud_decisions_total Decisions by action",
            "# TYPE fraud_decisions_total counter",
        ]
        for action in ("approve", "review", "decline"):
            lines.append(
                f'fraud_decisions_total{{action="{action}",model="{version}"}} {self.actions[action]}'
            )
        lines += ["# TYPE fraud_errors_total counter", f"fraud_errors_total {self.errors}"]
        if self.latencies:
            lat = np.array(self.latencies)
            lines.append("# TYPE fraud_latency_ms summary")
            for q in (0.5, 0.95, 0.99):
                lines.append(f'fraud_latency_ms{{quantile="{q}"}} {np.quantile(lat, q):.3f}')
        return "\n".join(lines) + "\n"


def create_app(
    bundle: ModelBundle | None = None,
    rules: RuleEngine | None = None,
    warmup_path: str | None = None,
) -> FastAPI:
    bundle = bundle or load_bundle(os.environ.get("FRAUD_BUNDLE", "artifacts"))
    if rules is None and os.environ.get("FRAUD_RULES"):
        rules = RuleEngine.from_yaml(os.environ["FRAUD_RULES"])
    engine = DecisionEngine(bundle, rules=rules)
    warmup_path = warmup_path or os.environ.get("FRAUD_WARMUP")
    if warmup_path:
        engine.store.warm_up(load_transactions([warmup_path]))

    # The feature store is mutable shared state; serialise access to it.
    lock = threading.Lock()
    metrics = ServiceMetrics()

    app = FastAPI(
        title="Fraud Detection API",
        version=bundle.version,
        description="Scores card transactions in real time and returns approve / review / decline.",
