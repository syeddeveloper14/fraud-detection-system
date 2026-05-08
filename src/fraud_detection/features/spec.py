"""Feature catalogue: the single source of truth for feature names, order and meaning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema import CATEGORIES

# Rolling windows over each card's history: name -> (pandas offset, seconds).
WINDOWS: dict[str, tuple[str, int]] = {
    "1h": ("1h", 3_600),
    "24h": ("24h", 86_400),
    "7d": ("7D", 604_800),
}
BASELINE_WINDOW: tuple[str, int] = ("30D", 2_592_000)
MIN_GAP_HOURS = 1 / 60  # floor for speed calculations (one minute)

STATIC_FEATURES = [
    "amt",
    "log_amt",
    "hour",
    "day_of_week",
    "is_night",
    "age_years",
    "distance_home_km",
    "log_city_pop",
]
HISTORY_FEATURES = [
    *[f"txn_count_{w}" for w in WINDOWS],
    *[f"amt_sum_{w}" for w in WINDOWS],
    "amt_mean_30d",
    "amt_std_30d",
    "amt_zscore_30d",
    "amt_ratio_30d",
    "secs_since_last",
    "km_from_last",
    "speed_kmh",
    "is_new_merchant",
    "n_prior_txns",
]
ENCODED_FEATURES = ["merchant_risk", "category_risk"]
CATEGORICAL_FEATURES = ["category"]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "amt": "Transaction amount",
    "log_amt": "log(1 + amount)",
    "hour": "Hour of day (0-23)",
    "day_of_week": "Day of week (Mon=0)",
    "is_night": "1 if hour is 22:00-05:59",
    "age_years": "Cardholder age at transaction time",
    "distance_home_km": "Distance from cardholder home to merchant",
    "log_city_pop": "log(1 + population of cardholder city)",
    "txn_count_1h": "Card transactions in the previous hour",
    "txn_count_24h": "Card transactions in the previous 24 hours",
    "txn_count_7d": "Card transactions in the previous 7 days",
    "amt_sum_1h": "Card spend in the previous hour",
    "amt_sum_24h": "Card spend in the previous 24 hours",
    "amt_sum_7d": "Card spend in the previous 7 days",
    "amt_mean_30d": "Mean card transaction amount over the previous 30 days",
    "amt_std_30d": "Std of card transaction amounts over the previous 30 days",
    "amt_zscore_30d": "(amt - mean_30d) / (std_30d + 1)",
    "amt_ratio_30d": "amt / (mean_30d + 1)",
    "secs_since_last": "Seconds since the card's previous transaction",
    "km_from_last": "Distance between this merchant and the previous one",
    "speed_kmh": "Implied travel speed from the previous transaction",
    "is_new_merchant": "1 if the card has never used this merchant before",
    "n_prior_txns": "Number of earlier transactions on the card",
    "merchant_risk": "Smoothed historical fraud rate of the merchant",
    "category_risk": "Smoothed historical fraud rate of the category",
    "category": "Merchant category (categorical)",
}


@dataclass
class FeatureSpec:
    """Column order and encoding of the model matrix.

    The matrix is all-float: categorical features are stored as integer codes (unknown or
    missing -> NaN), which lets the online path build a NumPy row without pandas.
    """

    numeric: list[str] = field(
        default_factory=lambda: [*STATIC_FEATURES, *HISTORY_FEATURES, *ENCODED_FEATURES]
    )
    categorical: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    category_levels: list[str] = field(default_factory=lambda: list(CATEGORIES))

    def __post_init__(self) -> None:
        self._codes = {level: float(i) for i, level in enumerate(self.category_levels)}

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    @property
    def categorical_mask(self) -> np.ndarray:
        return np.array([c in self.categorical for c in self.columns])

    def code(self, value: object) -> float:
        return self._codes.get(str(value), np.nan)

    def to_model_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fixed column order, float dtype, categoricals as codes."""
        missing = [c for c in self.columns if c not in features.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        frame = features[self.numeric].astype(float).replace([np.inf, -np.inf], np.nan)
        for col in self.categorical:
            frame[col] = features[col].astype(str).map(self._codes).astype(float)
        return frame

    def row_vector(self, values: dict[str, object]) -> np.ndarray:
        """Single-row matrix for online scoring (same layout as ``to_model_frame``)."""
        row = [float(values[c]) for c in self.numeric]
        row += [self.code(values[c]) for c in self.categorical]
        vec = np.asarray([row], dtype=float)
        vec[~np.isfinite(vec)] = np.nan
        return vec
