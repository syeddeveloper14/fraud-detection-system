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


