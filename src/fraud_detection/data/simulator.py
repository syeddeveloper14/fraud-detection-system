"""Synthetic card transactions in the Sparkov / Kaggle schema.

Legitimate behaviour: each card has its own spending rate and category mix, buys from merchants
near home, mostly during the day, with category-specific amounts.

Fraud is injected as whole episodes on a subset of cards, using three patterns seen in
real card fraud:

* ``card_testing``: a burst of tiny online charges, then a few large ones;
* ``account_takeover``: high-value night-time spending over one to three days;
* ``geo_jump``: purchases in a distant city shortly after a normal local purchase.

The goal is not realism in every detail; it is a dataset where velocity, time-of-day,
amount and distance features carry signal, so the full pipeline can be exercised and tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import CATEGORIES, RAW_COLUMNS

_FIRST = [
    "James",
    "Mary",
    "John",
    "Linda",
    "Robert",
    "Susan",
    "Michael",
    "Karen",
    "David",
    "Nancy",
    "Daniel",
    "Lisa",
    "Paul",
    "Sarah",
    "Mark",
    "Emily",
    "Steven",
    "Laura",
    "Kevin",
    "Amy",
]
_LAST = [
    "Smith",
    "Johnson",
    "Brown",
    "Garcia",
    "Miller",
    "Davis",
    "Lopez",
    "Wilson",
    "Moore",
    "Clark",
    "Lewis",
    "Walker",
    "Hall",
    "Young",
    "King",
    "Wright",
    "Scott",
    "Green",
    "Baker",
    "Adams",
]
_JOBS = [
    "Engineer",
    "Teacher",
    "Nurse",
    "Accountant",
    "Designer",
    "Electrician",
    "Pharmacist",
    "Chef",
    "Analyst",
    "Architect",
    "Paramedic",
    "Librarian",
]
_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "WA", "AZ", "MA", "CO", "OR"]

# Typical amount (median, log-sd) per category and a coarse hour-of-day profile.
_AMOUNT = {
    "entertainment": (40, 0.8),
    "food_dining": (35, 0.7),
    "gas_transport": (55, 0.5),
    "grocery_net": (45, 0.6),
    "grocery_pos": (75, 0.6),
    "health_fitness": (40, 0.7),
    "home": (60, 0.9),
    "kids_pets": (45, 0.8),
    "misc_net": (30, 1.0),
    "misc_pos": (35, 0.9),
    "personal_care": (25, 0.8),
    "shopping_net": (50, 1.0),
    "shopping_pos": (55, 0.9),
    "travel": (120, 1.1),
}
_BASE_MIX = np.array([5, 7, 10, 3, 9, 6, 9, 8, 4, 6, 7, 6, 8, 2], dtype=float)
_DAY_HOURS = np.array(
    [1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 10, 11, 12, 12, 11, 10, 10, 11, 12, 11, 9, 6, 3, 2], dtype=float
)


def _hex_ids(rng: np.random.Generator, n: int) -> list[str]:
    return [
        f"{a:016x}{b:016x}"
