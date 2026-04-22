"""Time-ordered train / validation / test split.

Random splits leak future behaviour of the same card into training and overstate
performance, so the split is strictly by timestamp.
"""

from __future__ import annotations

import pandas as pd


def time_split(
