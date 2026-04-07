"""Column names for raw transactions (Sparkov / Kaggle "Credit Card Transactions Fraud Detection")."""

from __future__ import annotations

RAW_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]

# The subset of fields the scoring service needs. Names, street address and job are never
# used as model inputs.
SCORING_FIELDS = [
