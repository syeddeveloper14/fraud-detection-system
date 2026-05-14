"""Features that depend only on the transaction itself (no card history).

Used unchanged by the offline pipeline and the online service.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import haversine_km


def static_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["ts"])
    dob = pd.to_datetime(df["dob"])
    hour = ts.dt.hour
    out = pd.DataFrame(index=df.index)
    out["amt"] = df["amt"].astype(float)
    out["log_amt"] = np.log1p(out["amt"])
    out["hour"] = hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)
    out["age_years"] = (ts - dob).dt.days / 365.25
    out["distance_home_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    out["log_city_pop"] = np.log1p(df["city_pop"].astype(float))
    out["category"] = df["category"].astype(str).to_numpy()
    return out


def static_row(txn: dict) -> dict[str, object]:
    """Single-transaction version of ``static_features`` without pandas frames.

    Must produce identical values; ``tests/test_feature_parity.py`` checks this.
    """
    ts = pd.Timestamp(txn["ts"])
    dob = pd.Timestamp(txn["dob"])
    amt = float(txn["amt"])
    hour = ts.hour
    return {
        "amt": amt,
        "log_amt": float(np.log1p(amt)),
        "hour": hour,
        "day_of_week": ts.dayofweek,
        "is_night": int(hour >= 22 or hour < 6),
        "age_years": (ts - dob).days / 365.25,
        "distance_home_km": float(
            haversine_km(txn["lat"], txn["long"], txn["merch_lat"], txn["merch_long"])
        ),
        "log_city_pop": float(np.log1p(float(txn["city_pop"]))),
        "category": str(txn["category"]),
    }
