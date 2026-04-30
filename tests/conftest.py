import pandas as pd
import pytest

from fraud_detection.config import load_config
from fraud_detection.data import prepare_transactions, simulate_transactions
from fraud_detection.pipeline import train_pipeline


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    return prepare_transactions(
        simulate_transactions(n_cards=200, days=75, fraud_card_share=0.35, seed=11)
    )


@pytest.fixture(scope="session")
def trained(tmp_path_factory, small_df):
    artifacts = tmp_path_factory.mktemp("artifacts")
    cfg = load_config(
        overrides={
            "artifacts_dir": str(artifacts),
            "model.max_iter": 150,
            "rules_path": "configs/rules.yaml",
        }
    )
    return cfg, train_pipeline(cfg, df=small_df)


def make_txn(**overrides) -> dict:
    base = {
        "trans_num": "t1",
        "cc_num": "4000000000000001",
        "ts": pd.Timestamp("2023-03-01 12:00:00"),
        "amt": 42.0,
        "merchant": "fraud_Merchant_0001",
        "category": "grocery_pos",
        "lat": 40.0,
        "long": -90.0,
        "merch_lat": 40.1,
        "merch_long": -90.1,
        "city_pop": 5000,
        "dob": pd.Timestamp("1980-05-17"),
    }
    base.update(overrides)
    return base
