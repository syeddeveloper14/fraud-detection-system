"""Online feature store: per-card state updated one transaction at a time.

Produces the same history features as ``history.history_features`` (verified by
``tests/test_feature_parity.py``) with O(events in the last 30 days) work per request.

The in-memory backend is enough for a single scoring process. ``StateBackend`` is the seam
for a shared store (Redis, DynamoDB) when the service is scaled horizontally.
"""

from __future__ import annotations

import math
import pickle
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .geo import haversine_km
from .spec import BASELINE_WINDOW, MIN_GAP_HOURS, WINDOWS


@dataclass
class CardState:
    events: deque = field(default_factory=deque)  # (ts_seconds, amt), oldest first
    last_ts: float | None = None
    last_lat: float | None = None
    last_long: float | None = None
    n_txns: int = 0
    merchants: set[str] = field(default_factory=set)


class StateBackend(Protocol):
    def get(self, card: str) -> CardState | None: ...
    def put(self, card: str, state: CardState) -> None: ...
    def __len__(self) -> int: ...
    def items(self) -> Iterator[tuple[str, CardState]]: ...


class InMemoryBackend:
    def __init__(self) -> None:
        self._data: dict[str, CardState] = {}

    def get(self, card: str) -> CardState | None:
        return self._data.get(card)

    def put(self, card: str, state: CardState) -> None:
