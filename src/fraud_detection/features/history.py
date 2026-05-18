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

