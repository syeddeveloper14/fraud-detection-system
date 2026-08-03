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
