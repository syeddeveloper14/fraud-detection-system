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
