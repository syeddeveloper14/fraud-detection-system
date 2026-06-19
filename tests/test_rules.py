import math

import pytest

from fraud_detection.rules import Rule, RuleEngine, RuleError


def engine():
    return RuleEngine.from_dict(
        {
            "rules": [
                {
                    "name": "burst",
                    "when": {
                        "all": [
                            {"feature": "txn_count_1h", "op": ">=", "value": 4},
                            {"feature": "amt", "op": "<", "value": 5},
                        ]
                    },
                    "action": "review",
                    "reason": "card testing",
                },
                {
                    "name": "watchlist",
                    "when": {"any": [{"feature": "merchant", "op": "in", "value": ["bad"]}]},
                    "action": "decline",
                },
                {
                    "name": "not_domestic_night",
                    "when": {"not": {"feature": "is_night", "op": "==", "value": 0}},
                    "action": "review",
                    "enabled": False,
                },
            ]
        }
    )


def test_all_any():
    e = engine()
    hits = e.evaluate({"txn_count_1h": 5, "amt": 1.0, "merchant": "bad"})
    assert {h.name for h in hits} == {"burst", "watchlist"}
