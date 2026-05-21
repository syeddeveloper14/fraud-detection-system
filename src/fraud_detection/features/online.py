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
        self._data[card] = state

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> Iterator[tuple[str, CardState]]:
        return iter(self._data.items())


def _to_seconds(ts: object) -> float:
    return pd.Timestamp(ts).value / 1e9


class OnlineFeatureStore:
    def __init__(self, backend: StateBackend | None = None) -> None:
        self.backend: StateBackend = backend or InMemoryBackend()
        self._max_window = max(BASELINE_WINDOW[1], *(s for _, s in WINDOWS.values()))

    # ----------------------------------------------------------------- features
    def compute(self, txn: dict) -> dict[str, float]:
        """History features for ``txn`` given everything seen so far (no state change)."""
        t = _to_seconds(txn["ts"])
        amt = float(txn["amt"])
        state = self.backend.get(str(txn["cc_num"]))
        feats: dict[str, float] = {}

        events = state.events if state else ()
        for name, (_, seconds) in WINDOWS.items():
            lo = t - seconds
            window_amts = [a for (et, a) in events if lo <= et < t]
            feats[f"txn_count_{name}"] = float(len(window_amts))
            feats[f"amt_sum_{name}"] = float(sum(window_amts))

        lo = t - BASELINE_WINDOW[1]
        base = np.array([a for (et, a) in events if lo <= et < t], dtype=float)
        mean = float(base.mean()) if len(base) else math.nan
        std = float(base.std(ddof=1)) if len(base) >= 2 else math.nan
        feats["amt_mean_30d"] = mean
        feats["amt_std_30d"] = std
        feats["amt_zscore_30d"] = (amt - mean) / ((0.0 if math.isnan(std) else std) + 1.0)
        feats["amt_ratio_30d"] = amt / (mean + 1.0)

        if state is not None and state.last_ts is not None:
            gap = t - state.last_ts
            km = float(
                haversine_km(state.last_lat, state.last_long, txn["merch_lat"], txn["merch_long"])
            )
            feats["secs_since_last"] = gap
            feats["km_from_last"] = km
            feats["speed_kmh"] = km / max(gap / 3600.0, MIN_GAP_HOURS)
        else:
