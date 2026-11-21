"""Reproducible synthetic inputs for the ANNA demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Item:
    name: str
    unit: str
    average_daily_demand: float
    shelf_life_days: int
    variability: float


ITEMS = (
    Item("Rice", "kg", 34.0, 365, 0.12),
    Item("Wheat flour", "kg", 28.0, 120, 0.16),
    Item("Pulses", "kg", 15.0, 180, 0.19),
    Item("Cooking oil", "litres", 9.0, 180, 0.14),
    Item("Packaged food", "packs", 22.0, 90, 0.22),
)


def generate_history(seed: int, days: int = 365) -> pd.DataFrame:
    """Generate one year of synthetic daily demand for each modeled item."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    rows: list[dict[str, object]] = []
    for item_index, item in enumerate(ITEMS):
        phase = item_index * 0.7
        for index, date in enumerate(dates):
            annual = 1.0 + 0.10 * np.sin(2 * np.pi * index / 365.0 + phase)
            weekly = 1.0 + 0.04 * np.sin(2 * np.pi * date.dayofweek / 7.0)
            noise = rng.normal(1.0, item.variability)
            spike = rng.uniform(1.25, 1.8) if rng.random() < 0.018 else 1.0
            demand = max(0.0, item.average_daily_demand * annual * weekly * noise * spike)
            rows.append({"date": date, "item": item.name, "daily_consumption": round(demand, 2)})
    return pd.DataFrame(rows)


def generate_batches(seed: int, history: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic batch stock, including varied expiry dates."""
    rng = np.random.default_rng(seed + 10_007)
    as_of = pd.Timestamp(history["date"].max()).normalize()
    rows: list[dict[str, object]] = []
    for item in ITEMS:
        series = history.loc[history.item == item.name, "daily_consumption"]
        average = float(series.tail(28).mean())
        total_days = float(rng.uniform(18, 32))
        portions = rng.dirichlet([2.2, 2.0, 1.6])
        # Two batches expire sooner, so FEFO and expiry exposure are visible in the demo.
        expiry_offsets = [max(8, min(item.shelf_life_days - 1, 24)),
                          max(20, min(item.shelf_life_days - 1, 58)),
                          max(35, item.shelf_life_days - 25)]
        for batch_no, (portion, offset) in enumerate(zip(portions, expiry_offsets), start=1):
            received = as_of - pd.Timedelta(days=int(rng.integers(15, 55)))
            rows.append({
                "batch_id": f"SIM-{item.name[:2].upper()}-{batch_no:02d}",
                "item": item.name,
                "quantity": round(average * total_days * float(portion), 2),
                "received_date": received,
                "expiry_date": as_of + pd.Timedelta(days=int(offset)),
            })
    return pd.DataFrame(rows)


def get_item(name: str) -> Item:
    """Return the configured item or raise a clear error for unknown names."""
    for item in ITEMS:
        if item.name == name:
            return item
    raise ValueError(f"Unknown item: {name}")
