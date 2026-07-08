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
    counts = values.astype(str).value_counts()
    known = np.array([counts.get(level, 0) for level in levels], dtype=float)
    other = len(values) - known.sum()
    return np.r_[known, other] / max(len(values), 1)


def reference_profile(
    X: pd.DataFrame,
    scores: np.ndarray | None = None,
    n_bins: int = 10,
    categorical: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    profile: dict[str, Any] = {"features": {}, "n": int(len(X))}
    for col in X.columns:
        if (
            col in categorical
            or isinstance(X[col].dtype, pd.CategoricalDtype)
            or X[col].dtype == object
        ):
            profile["features"][col] = _categorical_profile(X[col])
        else:
            profile["features"][col] = _numeric_profile(X[col], n_bins)
    if scores is not None:
        profile["score"] = _numeric_profile(pd.Series(scores), n_bins)
    return profile


def drift_report(
    profile: dict[str, Any],
    X: pd.DataFrame,
    scores: np.ndarray | None = None,
    warn: float = 0.1,
    alert: float = 0.25,
) -> pd.DataFrame:
    rows = []
    items = list(profile["features"].items())
    if scores is not None and "score" in profile:
        items.append(("__score__", profile["score"]))
    for name, ref in items:
        values = pd.Series(scores) if name == "__score__" else X[name]
        if ref["kind"] == "numeric":
            current = _numeric_props(values.to_numpy(dtype=float), ref["edges"])
            expected = np.asarray(ref["proportions"])
        else:
            current = _categorical_props(values, ref["levels"])
            expected = np.r_[ref["proportions"], 0.0]
        value = psi(expected, current)
        status = "alert" if value >= alert else "warn" if value >= warn else "ok"
        rows.append({"feature": name, "psi": value, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
