"""Declarative rules evaluated next to the model.

Rules encode knowledge the model cannot learn quickly (a new attack pattern, a merchant under
investigation) and give operations a lever that does not require retraining. They are written
in YAML and evaluated with a small, safe interpreter (no ``eval``).

Example::

    rules:
      - name: card_testing_burst
        when:
          all:
            - {feature: txn_count_1h, op: ">=", value: 4}
            - {feature: amt, op: "<", value: 5}
        action: review
        reason: Burst of small charges (possible card testing)
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
}
RULE_ACTIONS = ("review", "decline")
SEVERITY = {"approve": 0, "review": 1, "decline": 2}


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    name: str
    when: dict[str, Any]
    action: str
    reason: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class RuleHit:
    name: str
    action: str
    reason: str


def _validate(cond: dict[str, Any], path: str) -> None:
    if not isinstance(cond, dict) or not cond:
        raise RuleError(f"{path}: condition must be a non-empty mapping")
    if "all" in cond or "any" in cond:
        key = "all" if "all" in cond else "any"
        children = cond[key]
        if not isinstance(children, list) or not children:
            raise RuleError(f"{path}.{key}: expected a non-empty list")
        for i, child in enumerate(children):
            _validate(child, f"{path}.{key}[{i}]")
    elif "not" in cond:
        _validate(cond["not"], f"{path}.not")
    else:
        missing = {"feature", "op", "value"} - set(cond)
        if missing:
            raise RuleError(f"{path}: leaf condition missing {sorted(missing)}")
        if cond["op"] not in OPERATORS:
            raise RuleError(f"{path}: unknown operator {cond['op']!r}")


def _evaluate(cond: dict[str, Any], ctx: dict[str, Any]) -> bool:
    if "all" in cond:
        return all(_evaluate(c, ctx) for c in cond["all"])
    if "any" in cond:
        return any(_evaluate(c, ctx) for c in cond["any"])
    if "not" in cond:
        return not _evaluate(cond["not"], ctx)
    value = ctx.get(cond["feature"])
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False  # missing data never triggers a rule
    try:
        return bool(OPERATORS[cond["op"]](value, cond["value"]))
    except TypeError:
        return False


class RuleEngine:
    def __init__(self, rules: list[Rule]) -> None:
        names = [r.name for r in rules]
        if len(names) != len(set(names)):
            raise RuleError("Rule names must be unique")
        for rule in rules:
            if rule.action not in RULE_ACTIONS:
                raise RuleError(f"{rule.name}: action must be one of {RULE_ACTIONS}")
            _validate(rule.when, rule.name)
        self.rules = rules

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleEngine:
        return cls([Rule(**r) for r in data.get("rules", [])])

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuleEngine:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    @classmethod
    def empty(cls) -> RuleEngine:
        return cls([])

    def evaluate(self, context: dict[str, Any]) -> list[RuleHit]:
        return [
            RuleHit(r.name, r.action, r.reason)
            for r in self.rules
            if r.enabled and _evaluate(r.when, context)
        ]

    def __len__(self) -> int:
        return len(self.rules)
