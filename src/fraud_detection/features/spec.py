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
