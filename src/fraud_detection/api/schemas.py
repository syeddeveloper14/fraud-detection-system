"""Request / response models for the scoring API."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    trans_num: str = Field(..., min_length=1, max_length=64)
    cc_num: str = Field(..., min_length=4, max_length=32)
    ts: datetime
    amt: float = Field(..., gt=0, lt=1_000_000)
    merchant: str
    category: str
    lat: float = Field(..., ge=-90, le=90)
    long: float = Field(..., ge=-180, le=180)
    merch_lat: float = Field(..., ge=-90, le=90)
    merch_long: float = Field(..., ge=-180, le=180)
    city_pop: int = Field(..., ge=0)
    dob: date

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trans_num": "0b242abb623afc578575680df30655b9",
                    "cc_num": "4613314721966",
                    "ts": "2023-06-21T23:14:05",
                    "amt": 812.47,
                    "merchant": "fraud_Merchant_0042",
                    "category": "shopping_net",
                    "lat": 40.4931,
                    "long": -91.8912,
                    "merch_lat": 41.1012,
                    "merch_long": -92.2210,
