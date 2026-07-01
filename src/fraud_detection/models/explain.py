"""Global importance and per-transaction reason codes.

* Global: permutation importance measured as the drop in average precision.
* Local: occlusion. For each of the most important features, replace the transaction's value
  with a "typical legitimate" baseline and re-score. Features whose replacement lowers the
  score most are returned as reason codes. Model-agnostic and cheap (one batched predict).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..features.spec import FEATURE_DESCRIPTIONS
from .model import FraudModel


def permutation_importance(
    model: FraudModel,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 3,
    max_rows: int = 20_000,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if len(X) > max_rows:
        # Keep every fraud row; sample legitimate rows.
        pos = np.flatnonzero(y == 1)
        neg = rng.choice(np.flatnonzero(y == 0), size=max_rows - len(pos), replace=False)
        idx = np.sort(np.r_[pos, neg])
        X, y = X.iloc[idx], y[idx]
    base = average_precision_score(y, model.predict_raw_matrix(X))
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[col] = (
                X[col].sample(frac=1.0, random_state=int(rng.integers(1 << 31))).to_numpy()
            )
            drops.append(base - average_precision_score(y, model.predict_raw_matrix(shuffled)))
        rows.append(
            {"feature": col, "importance": float(np.mean(drops)), "std": float(np.std(drops))}
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def legit_baseline(
    X: pd.DataFrame, y: np.ndarray, categorical: list[str] | tuple[str, ...] = ()
) -> dict[str, float]:
    """Typical legitimate value per feature: median for numeric, mode for categorical codes."""
    legit = X[np.asarray(y) == 0]
    return {
        col: float(legit[col].mode().iat[0]) if col in categorical else float(legit[col].median())
        for col in X.columns
    }


@dataclass
class ReasonCoder:
    baseline: dict[str, float]
    candidates: list[str] = field(default_factory=list)  # features to test, most important first
    category_levels: dict[str, list[str]] = field(default_factory=dict)  # decode codes for display
    top_k: int = 3

    def _display(self, col: str, value: float) -> float | str | None:
        if pd.isna(value):
            return None
        if col in self.category_levels:
            levels = self.category_levels[col]
            return levels[int(value)] if 0 <= int(value) < len(levels) else None
        return float(value)

    def explain(self, model: FraudModel, X_row: pd.DataFrame) -> list[dict[str, object]]:
        if len(X_row) != 1:
            raise ValueError("explain() expects a single-row model frame")
        cols = [c for c in self.candidates if c in X_row.columns]
        if not cols:
            return []
        batch = np.repeat(X_row.to_numpy(dtype=float), len(cols) + 1, axis=0)
        positions = {c: i for i, c in enumerate(X_row.columns)}
        for i, col in enumerate(cols, start=1):
            batch[i, positions[col]] = self.baseline[col]
        scores = model.predict_raw_matrix(batch)
        impact = scores[0] - scores[1:]
        reasons = []
        for j in np.argsort(-impact)[: self.top_k]:
            if impact[j] <= 0:
                break
            col = cols[j]
            reasons.append(
                {
                    "feature": col,
                    "description": FEATURE_DESCRIPTIONS.get(col, col),
                    "value": self._display(col, X_row.iloc[0][col]),
                    "impact": float(impact[j]),
                }
            )
        return reasons
