"""Population Stability Index (PSI) drift monitoring.

A reference profile (bin edges + proportions per feature, plus the score distribution) is
stored inside the model bundle at training time, so production monitoring does not need
access to the training data.

Rule of thumb: PSI < 0.1 stable, 0.1-0.25 investigate, > 0.25 significant shift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_profile(values: pd.Series, n_bins: int) -> dict[str, Any]:
    v = values.to_numpy(dtype=float)
    finite = v[~np.isnan(v)]
    if len(finite) == 0:
        edges: list[float] = []
    else:
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1))[1:-1]).tolist()
    return {"kind": "numeric", "edges": edges, "proportions": _numeric_props(v, edges).tolist()}


def _numeric_props(v: np.ndarray, edges: list[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    nan = np.isnan(v)
    counts = np.bincount(np.searchsorted(edges, v[~nan], side="right"), minlength=len(edges) + 1)
    counts = np.r_[counts, nan.sum()]  # last bucket holds missing values
    return counts / max(len(v), 1)


def _categorical_profile(values: pd.Series) -> dict[str, Any]:
    props = values.astype(str).value_counts(normalize=True)
    return {"kind": "categorical", "levels": props.index.tolist(), "proportions": props.tolist()}


def _categorical_props(values: pd.Series, levels: list[str]) -> np.ndarray:
