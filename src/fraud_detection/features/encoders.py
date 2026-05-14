"""Smoothed target encoding with out-of-fold values for training rows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


@dataclass
class TargetEncoder:
    """Encodes a high-cardinality key as a smoothed fraud rate.

    value(key) = (frauds(key) + m * prior) / (count(key) + m)

    Training rows get out-of-fold values so a row's own label never feeds its encoding.
    """

    smoothing: float = 50.0
    n_folds: int = 5
    seed: int = 42
    prior_: float = 0.0
    mapping_: dict[str, float] = field(default_factory=dict)

    def _table(self, keys: pd.Series, y: pd.Series) -> tuple[pd.Series, float]:
        prior = float(y.mean())
        stats = (
            pd.DataFrame({"k": keys.astype(str).to_numpy(), "y": y.to_numpy()})
            .groupby("k")["y"]
            .agg(["sum", "count"])
        )
        values = (stats["sum"] + self.smoothing * prior) / (stats["count"] + self.smoothing)
        return values, prior

    def fit(self, keys: pd.Series, y: pd.Series) -> TargetEncoder:
        values, self.prior_ = self._table(keys, y)
        self.mapping_ = values.to_dict()
        return self

    def transform(self, keys: pd.Series | list) -> np.ndarray:
        keys = pd.Series(keys).astype(str)
        return keys.map(self.mapping_).fillna(self.prior_).to_numpy(dtype=float)

    def transform_one(self, key: object) -> float:
        return float(self.mapping_.get(str(key), self.prior_))

    def fit_transform_oof(self, keys: pd.Series, y: pd.Series) -> np.ndarray:
        keys = keys.reset_index(drop=True).astype(str)
        y = y.reset_index(drop=True)
        oof = np.empty(len(keys))
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for fit_idx, apply_idx in folds.split(keys):
            values, prior = self._table(keys.iloc[fit_idx], y.iloc[fit_idx])
            oof[apply_idx] = keys.iloc[apply_idx].map(values).fillna(prior).to_numpy(dtype=float)
        self.fit(keys, y)
        return oof
