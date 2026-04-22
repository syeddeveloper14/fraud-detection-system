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
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. See data/README.md, or use data.source=synthetic."
            )
        frame = pd.read_csv(path, dtype={"cc_num": str, "zip": str, "trans_num": str})
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith("Unnamed")])
        frames.append(frame)
    return prepare_transactions(pd.concat(frames, ignore_index=True))


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, parse timestamps and sort into stream order.

    Stream order (timestamp, then trans_num) is the order the online feature store sees
    events in. Offline features use the same order so the two paths agree exactly.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    out["ts"] = pd.to_datetime(out["trans_date_trans_time"])
    out["dob"] = pd.to_datetime(out["dob"])
    out["cc_num"] = out["cc_num"].astype(str)
    out["trans_num"] = out["trans_num"].astype(str)
    out["amt"] = out["amt"].astype(float)
    if LABEL in out.columns:
        out[LABEL] = out[LABEL].astype(int)
    out = out.drop_duplicates("trans_num")
    return out.sort_values(["ts", "trans_num"], kind="mergesort").reset_index(drop=True)
