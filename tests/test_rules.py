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
    assert e.evaluate({"txn_count_1h": 5, "amt": 50.0, "merchant": "ok"}) == []


def test_disabled_rule_is_skipped():
    assert all(h.name != "not_domestic_night" for h in engine().evaluate({"is_night": 1}))


def test_missing_or_nan_never_fires():
    e = engine()
    assert e.evaluate({"amt": 1.0}) == []
    assert e.evaluate({"txn_count_1h": math.nan, "amt": 1.0}) == []


def test_type_mismatch_is_false():
    e = RuleEngine([Rule("x", {"feature": "amt", "op": ">", "value": 5}, "review")])
    assert e.evaluate({"amt": "abc"}) == []


@pytest.mark.parametrize(
    "rule",
    [
        Rule("bad_op", {"feature": "amt", "op": "~=", "value": 1}, "review"),
        Rule("bad_action", {"feature": "amt", "op": ">", "value": 1}, "approve"),
        Rule("bad_leaf", {"feature": "amt", "op": ">"}, "review"),
        Rule("empty_all", {"all": []}, "review"),
    ],
)
def test_validation(rule):
    with pytest.raises(RuleError):
        RuleEngine([rule])


def test_unique_names():
    r = Rule("dup", {"feature": "amt", "op": ">", "value": 1}, "review")
    with pytest.raises(RuleError):
        RuleEngine([r, r])


def test_shipped_rules_file_loads():
    e = RuleEngine.from_yaml("configs/rules.yaml")
    assert len(e) >= 3
    hits = e.evaluate({"speed_kmh": 2000.0, "km_from_last": 1500.0})
    assert [h.name for h in hits] == ["impossible_travel"]
