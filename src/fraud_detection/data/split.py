"""Time-ordered train / validation / test split.

Random splits leak future behaviour of the same card into training and overstate
performance, so the split is strictly by timestamp.
"""

from __future__ import annotations

import pandas as pd


def time_split(
    df: pd.DataFrame, valid_fraction: float = 0.15, test_fraction: float = 0.2, ts_col: str = "ts"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < valid_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("fractions must be in (0, 1)")
    if valid_fraction + test_fraction >= 1:
        raise ValueError("valid_fraction + test_fraction must be < 1")
    ordered = df.sort_values(ts_col, kind="mergesort")
    t_valid = ordered[ts_col].quantile(1 - valid_fraction - test_fraction)
    t_test = ordered[ts_col].quantile(1 - test_fraction)
    train = ordered[ordered[ts_col] < t_valid]
    valid = ordered[(ordered[ts_col] >= t_valid) & (ordered[ts_col] < t_test)]
    test = ordered[ordered[ts_col] >= t_test]
    return train, valid, test
