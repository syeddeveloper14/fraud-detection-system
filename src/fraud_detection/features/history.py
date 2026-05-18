"""Point-in-time card-history features computed in batch.

Definitions (mirrored exactly by ``online.OnlineFeatureStore``):

* window features use earlier events with timestamp in [t - W, t). Events with the same
  timestamp as the current one are excluded;
* "last event" features use the previous event in stream order (timestamp, then trans_num);
* ``is_new_merchant`` / ``n_prior_txns`` count events earlier in stream order.

Nothing at or after the current event can leak into its features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import haversine_km
from .spec import BASELINE_WINDOW, MIN_GAP_HOURS, WINDOWS


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return history features aligned to ``df.index``. ``df`` must be in stream order."""
    d = df[["cc_num", "ts", "amt", "merchant", "merch_lat", "merch_long"]].copy()
    d["_order"] = np.arange(len(d))
    d = d.sort_values(["cc_num", "_order"], kind="mergesort")

    out = pd.DataFrame(index=d.index)
    indexed = d.set_index("ts")
    grouped = indexed.groupby("cc_num", sort=False)["amt"]

    for name, (offset, _) in WINDOWS.items():
        rolling = grouped.rolling(offset, closed="left")
        out[f"txn_count_{name}"] = rolling.count().fillna(0).to_numpy()
        out[f"amt_sum_{name}"] = rolling.sum().fillna(0).to_numpy()

    baseline = grouped.rolling(BASELINE_WINDOW[0], closed="left")
    mean = baseline.mean().to_numpy()
    std = baseline.std().to_numpy()
    amt = d["amt"].to_numpy()
    out["amt_mean_30d"] = mean
    out["amt_std_30d"] = std
    out["amt_zscore_30d"] = (amt - mean) / (np.nan_to_num(std, nan=0.0) + 1.0)
    out["amt_ratio_30d"] = amt / (mean + 1.0)

    by_card = d.groupby("cc_num", sort=False)
    prev_ts = by_card["ts"].shift(1)
    gap_secs = (d["ts"] - prev_ts).dt.total_seconds()
    km = haversine_km(
        by_card["merch_lat"].shift(1),
        by_card["merch_long"].shift(1),
        d["merch_lat"],
        d["merch_long"],
    )
    out["secs_since_last"] = gap_secs.to_numpy()
    out["km_from_last"] = km
    out["speed_kmh"] = km / np.maximum(gap_secs.to_numpy() / 3600.0, MIN_GAP_HOURS)
    out["is_new_merchant"] = (
        (d.groupby(["cc_num", "merchant"], sort=False).cumcount() == 0).astype(int).to_numpy()
    )
    out["n_prior_txns"] = by_card.cumcount().to_numpy()

    return out.loc[df.index]
