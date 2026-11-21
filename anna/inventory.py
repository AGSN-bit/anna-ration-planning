"""Batch-aware stock logic and a compact disruption simulator."""

from __future__ import annotations

import pandas as pd


def allocate_fefo(batches: pd.DataFrame, demand: float, as_of: pd.Timestamp) -> tuple[pd.DataFrame, float, float]:
    """Consume the earliest-expiring eligible batches first without negative stock."""
    remaining = batches.copy()
    remaining["allocated"] = 0.0
    need = max(0.0, float(demand))
    eligible = remaining[(remaining.expiry_date >= as_of) & (remaining.quantity > 0)].sort_values(
        ["expiry_date", "received_date", "batch_id"]
    )
    for index in eligible.index:
        take = min(need, float(remaining.at[index, "quantity"]))
        remaining.at[index, "quantity"] = round(float(remaining.at[index, "quantity"]) - take, 6)
        remaining.at[index, "allocated"] = take
        need -= take
        if need <= 1e-9:
            break
    return remaining, max(0.0, float(demand) - need), need


def inventory_summary(batches: pd.DataFrame, as_of: pd.Timestamp, alert_days: int = 30) -> dict[str, float]:
    valid = batches[batches.expiry_date >= as_of]
    expired = batches[batches.expiry_date < as_of]
    near = valid[valid.expiry_date <= as_of + pd.Timedelta(days=alert_days)]
    return {
        "on_hand": float(batches.quantity.sum()),
        "usable": float(valid.quantity.sum()),
        "expired": float(expired.quantity.sum()),
        "near_expiry": float(near.quantity.sum()),
    }


def recommend_order(forecast_values, batches, as_of, lead_time: int, horizon: int, z: float = 1.28) -> dict[str, float]:
    demand = [float(x) for x in forecast_values]
    mean_daily = sum(demand[:min(lead_time, len(demand))]) / max(1, min(lead_time, len(demand)))
    std_daily = max(0.0, float(pd.Series(demand).std(ddof=1) or 0.0))
    safety = z * std_daily * (lead_time ** 0.5)
    lead_demand = mean_daily * lead_time
    rop = lead_demand + safety
    usable = inventory_summary(batches, as_of)["usable"]
    horizon_demand = sum(demand[:min(horizon, len(demand))])
    order_qty = max(0.0, horizon_demand + safety - usable)
    return {"mean_daily": mean_daily, "lead_demand": lead_demand, "safety_stock": safety,
            "reorder_point": rop, "usable_stock": usable, "horizon_demand": horizon_demand,
            "order_qty": order_qty, "needs_order": usable <= rop or order_qty > 0}


def simulate(batches: pd.DataFrame, forecast_values, start: pd.Timestamp, lead_time: int,
             delivery_delay: int, demand_multiplier: float, interruption_days: int,
             order_qty: float = 0.0) -> pd.DataFrame:
    """Replay forecast demand and an optional order under one disruption scenario."""
    stock = batches[["batch_id", "item", "quantity", "received_date", "expiry_date"]].copy()
    rows = []
    arrival_day = lead_time + delivery_delay
    for day, forecast_demand in enumerate(forecast_values):
        date = pd.Timestamp(start) + pd.Timedelta(days=day + 1)
        if order_qty > 0 and day == arrival_day:
            stock.loc[len(stock)] = [f"SIM-NEW-{date:%Y%m%d}", stock.item.iloc[0], order_qty,
                                      date, date + pd.Timedelta(days=365)]
        day_demand = float(forecast_demand) * demand_multiplier
        if day < interruption_days:
            day_demand *= 1.15
        stock, served, shortage = allocate_fefo(stock, day_demand, date)
        stock = stock[stock.expiry_date >= date].copy()
        rows.append({"date": date, "demand": day_demand, "served": served, "shortage": shortage,
                     "closing_stock": float(stock.quantity.sum()),
                     "usable_stock": inventory_summary(stock, date)["usable"]})
    return pd.DataFrame(rows)
