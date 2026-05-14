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
