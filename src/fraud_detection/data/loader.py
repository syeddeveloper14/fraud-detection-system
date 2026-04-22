"""Load raw transaction files and normalise them for feature engineering."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..schema import LABEL

REQUIRED = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "lat",
    "long",
    "city_pop",
    "dob",
    "trans_num",
    "merch_lat",
    "merch_long",
]


def load_transactions(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
