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
        for a, b in zip(rng.integers(0, 2**63, n), rng.integers(0, 2**63, n), strict=True)
    ]


def _hours(rng: np.random.Generator, n: int) -> np.ndarray:
    p = _DAY_HOURS / _DAY_HOURS.sum()
    return rng.choice(24, size=n, p=p) + rng.random(n)


def simulate_transactions(
    n_cards: int = 1000,
    days: int = 180,
    start: str = "2023-01-01",
    n_merchants: int = 400,
    fraud_card_share: float = 0.12,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)

    # Cities and cardholders -------------------------------------------------
    n_cities = 60
    city_lat = rng.uniform(26, 47, n_cities)
    city_long = rng.uniform(-122, -72, n_cities)
    city_pop = np.exp(rng.normal(9.5, 1.6, n_cities)).astype(int) + 500
    card_city = rng.integers(0, n_cities, n_cards)
    home_lat = city_lat[card_city] + rng.normal(0, 0.05, n_cards)
    home_long = city_long[card_city] + rng.normal(0, 0.05, n_cards)
    cc_nums = [f"4{int(x):015d}" for x in rng.integers(10**14, 10**15 - 1, n_cards)]
    ages_days = rng.uniform(18 * 365, 85 * 365, n_cards)
    dobs = [(start_ts - pd.Timedelta(days=float(a))).normalize() for a in ages_days]
    rate = rng.gamma(2.0, 0.6, n_cards)  # transactions per day
    mixes = np.vstack([rng.dirichlet(_BASE_MIX) for _ in range(n_cards)])

    merchant_cat = rng.integers(0, len(CATEGORIES), n_merchants)
    merchants_by_cat = {
        c: [f"fraud_Merchant_{m:04d}" for m in np.flatnonzero(merchant_cat == i)]
        or [f"fraud_Merchant_{n_merchants + i:04d}"]
        for i, c in enumerate(CATEGORIES)
    }
    # Flattened lookup so merchant choice can be vectorised per category.
    flat_names = np.array([m for c in CATEGORIES for m in merchants_by_cat[c]])
    sizes = np.array([len(merchants_by_cat[c]) for c in CATEGORIES])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    log_median = np.log([_AMOUNT[c][0] for c in CATEGORIES])
    log_sd = np.array([_AMOUNT[c][1] for c in CATEGORIES])
    start64 = np.datetime64(start_ts.to_datetime64(), "s")

    # Legitimate activity (vectorised per card) -------------------------------
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("card", "ts", "cat", "amt", "mlat", "mlong")
    }
    for card in range(n_cards):
        counts = rng.poisson(rate[card], days)
        n = int(counts.sum())
        if n == 0:
            continue
        seconds = np.repeat(np.arange(days), counts) * 86400 + (_hours(rng, n) * 3600).astype(int)
        cats = rng.choice(len(CATEGORIES), size=n, p=mixes[card])
        parts["card"].append(np.full(n, card))
        parts["ts"].append(start64 + seconds.astype("timedelta64[s]"))
        parts["cat"].append(cats)
        parts["amt"].append(np.exp(rng.normal(log_median[cats], log_sd[cats])))
        parts["mlat"].append(home_lat[card] + rng.uniform(-0.8, 0.8, n))
        parts["mlong"].append(home_long[card] + rng.uniform(-0.8, 0.8, n))
    legit_cats = np.concatenate(parts["cat"])
    pick = (rng.random(len(legit_cats)) * sizes[legit_cats]).astype(int)
    legit = pd.DataFrame(
        {
            "card": np.concatenate(parts["card"]),
            "ts": pd.to_datetime(np.concatenate(parts["ts"])),
            "category": np.array(CATEGORIES)[legit_cats],
            "amt": np.round(np.maximum(np.concatenate(parts["amt"]), 1.0), 2),
            "merchant": flat_names[offsets[legit_cats] + pick],
            "merch_lat": np.concatenate(parts["mlat"]),
            "merch_long": np.concatenate(parts["mlong"]),
            "is_fraud": 0,
        }
    )

    rows: list[dict] = []

    def add(
        card: int,
        ts: pd.Timestamp,
        category: str,
        amt: float,
        m_lat: float,
        m_long: float,
        fraud: int,
    ) -> None:
        names = merchants_by_cat[category]
        rows.append(
            {
                "card": card,
                "ts": ts,
                "category": category,
                "amt": round(float(max(amt, 1.0)), 2),
                "merchant": names[rng.integers(len(names))],
                "merch_lat": m_lat,
                "merch_long": m_long,
                "is_fraud": fraud,
            }
        )

    # Fraud episodes -----------------------------------------------------------
    n_fraud_cards = max(1, int(round(fraud_card_share * n_cards)))
    fraud_cards = rng.choice(n_cards, size=n_fraud_cards, replace=False)
    for card in fraud_cards:
