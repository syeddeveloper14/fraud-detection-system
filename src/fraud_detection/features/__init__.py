from .encoders import TargetEncoder
from .geo import haversine_km
from .history import history_features
from .online import OnlineFeatureStore
from .spec import FEATURE_DESCRIPTIONS, FeatureSpec
from .static import static_features, static_row


def build_offline_features(df):
    """Static + history features (plus the merchant key) for a stream-ordered frame."""
    import pandas as pd

    feats = pd.concat([static_features(df), history_features(df)], axis=1)
    feats["merchant"] = df["merchant"].astype(str).to_numpy()
    return feats


__all__ = [
    "FEATURE_DESCRIPTIONS",
    "FeatureSpec",
    "OnlineFeatureStore",
    "TargetEncoder",
    "build_offline_features",
    "haversine_km",
    "history_features",
    "static_features",
    "static_row",
]
