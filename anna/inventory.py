"""Inventory, replenishment, FEFO, and policy simulation calculations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _finite_nonnegative(values: Iterable[float], label: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite daily values.")
    return np.maximum(array, 0.0)


def apply_demand_scenario(
    baseline_forecast: Iterable[float],
    demand_multiplier: float = 1.0,
    interruption_days: int = 0,
    interruption_multiplier: float = 1.15,
) -> np.ndarray:
    """Return scenario demand separately; interruption is an explicit demand-pressure proxy."""
    baseline = _finite_nonnegative(baseline_forecast, "Baseline forecast")
    if not np.isfinite(demand_multiplier) or demand_multiplier < 0:
        raise ValueError("Demand multiplier must be a finite, nonnegative number.")
    if not np.isfinite(interruption_multiplier) or interruption_multiplier < 0:
        raise ValueError("Interruption multiplier must be a finite, nonnegative number.")
    adjusted = baseline * float(demand_multiplier)
    days = max(0, int(interruption_days))
    adjusted[:days] *= float(interruption_multiplier)
    return adjusted


def allocate_fefo(
    batches: pd.DataFrame,
    demand: float,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, float, float]:
    """Allocate demand to usable batches by expiry, then receipt date and ID.

    Rows expiring before ``as_of`` are never consumed. Returns remaining batches,
    quantity served, and unmet demand.
    """
    required = {"batch_id", "quantity", "received_date", "expiry_date"}
    missing = required - set(batches.columns)
    if missing:
        raise ValueError(f"Batch table is missing columns: {', '.join(sorted(missing))}")
    if not np.isfinite(demand):
        raise ValueError("Demand must be finite.")
    remaining = batches.copy()
    remaining["allocated"] = 0.0
    need = max(0.0, float(demand))
    eligible = remaining[(remaining.expiry_date >= as_of) & (remaining.quantity > 0)].sort_values(
        ["expiry_date", "received_date", "batch_id"]
    )
    for index in eligible.index:
        take = min(need, float(remaining.at[index, "quantity"]))
        remaining.at[index, "quantity"] = max(0.0, float(remaining.at[index, "quantity"]) - take)
        remaining.at[index, "allocated"] = take
        need -= take
        if need <= 1e-9:
            need = 0.0
            break
    return remaining, max(0.0, float(demand) - need), need


def inventory_summary(
    batches: pd.DataFrame,
    as_of: pd.Timestamp,
    alert_days: int = 30,
) -> dict[str, float]:
    """Summarize stock without treating expired quantity as usable inventory."""
    valid = batches[batches.expiry_date >= as_of]
    expired = batches[batches.expiry_date < as_of]
    near = valid[valid.expiry_date <= as_of + pd.Timedelta(days=alert_days)]
    return {
        "on_hand": float(batches.quantity.clip(lower=0).sum()),
        "usable": float(valid.quantity.clip(lower=0).sum()),
        "expired": float(expired.quantity.clip(lower=0).sum()),
        "near_expiry": float(near.quantity.clip(lower=0).sum()),
    }


def recommend_order(
    forecast_values: Iterable[float],
    batches: pd.DataFrame,
    as_of: pd.Timestamp,
    lead_time_days: int,
    horizon_days: int,
    daily_demand_std: float,
    service_factor: float = 1.28,
    confirmed_incoming: float = 0.0,
) -> dict[str, float | int | bool]:
    """Calculate an illustrative order-up-to quantity from scenario demand.

    Reorder point = forecast demand through lead time + safety stock.
    Safety stock = service factor × historical daily-demand standard deviation
    × sqrt(lead time). Target coverage is the longer of the lead time and
    selected planning horizon. Expired stock is excluded; confirmed incoming
    stock is included in inventory position.
    """
    demand = _finite_nonnegative(forecast_values, "Forecast")
    if demand.size == 0:
        raise ValueError("At least one forecast day is required.")
    if lead_time_days < 1 or horizon_days < 1:
        raise ValueError("Lead time and planning horizon must be at least one day.")
    if not np.isfinite(daily_demand_std) or daily_demand_std < 0:
        raise ValueError("Daily demand variability must be finite and nonnegative.")
    if not np.isfinite(service_factor) or service_factor < 0:
        raise ValueError("Service factor must be finite and nonnegative.")
    if not np.isfinite(confirmed_incoming) or confirmed_incoming < 0:
        raise ValueError("Confirmed incoming stock must be finite and nonnegative.")

    # If a requested coverage window exceeds the supplied forecast, extend it
    # using the mean of the available forecast rather than silently truncating.
    mean_daily = float(demand.mean())
    needed_days = max(int(lead_time_days), int(horizon_days))
    extended = np.pad(demand, (0, max(0, needed_days - len(demand))), constant_values=mean_daily)
    lead_demand = float(extended[:lead_time_days].sum())
    horizon_demand = float(extended[:horizon_days].sum())
    coverage_demand = float(extended[:needed_days].sum())
    safety_stock = float(service_factor * daily_demand_std * np.sqrt(lead_time_days))
    reorder_point = lead_demand + safety_stock
    usable = inventory_summary(batches, as_of)["usable"]
    inventory_position = usable + float(confirmed_incoming)
    target_stock = coverage_demand + safety_stock
    order_qty = max(0.0, target_stock - inventory_position)
    return {
        "mean_daily": mean_daily,
        "lead_demand": lead_demand,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "usable_stock": usable,
        "confirmed_incoming": float(confirmed_incoming),
        "inventory_position": inventory_position,
        "horizon_demand": horizon_demand,
        "coverage_days": needed_days,
        "target_stock": target_stock,
        "order_qty": order_qty,
        "needs_order": inventory_position <= reorder_point or order_qty > 1e-9,
    }


def simulate_policy(
    batches: pd.DataFrame,
    daily_demand: Iterable[float],
    start: pd.Timestamp,
    horizon_days: int,
    lead_time_days: int,
    additional_delay_days: int = 0,
    order_qty: float = 0.0,
    order_shelf_life_days: int = 90,
) -> pd.DataFrame:
    """Replay daily demand with FEFO, optional pipeline order, and expiry waste.

    Day 1 is ``start + 1 day``. An order arrives before that day's demand on
    ``start + lead_time + additional_delay``. Orders arriving after the horizon
    remain in the pipeline and do not affect projected stock or shortages.
    """
    demand = _finite_nonnegative(daily_demand, "Daily demand")
    if horizon_days < 1:
        raise ValueError("Planning horizon must be at least one day.")
    if lead_time_days < 1 or additional_delay_days < 0:
        raise ValueError("Lead time must be positive and additional delay nonnegative.")
    if not np.isfinite(order_qty) or order_qty < 0:
        raise ValueError("Order quantity must be finite and nonnegative.")
    if order_shelf_life_days < 1:
        raise ValueError("Order shelf life must be at least one day.")
    if "item" not in batches.columns or batches.empty:
        raise ValueError("At least one item batch is required for simulation.")

    horizon = int(horizon_days)
    demand = np.pad(demand[:horizon], (0, max(0, horizon - len(demand))), constant_values=0.0)
    stock = batches[["batch_id", "item", "quantity", "received_date", "expiry_date"]].copy().reset_index(drop=True)
    lead_days = int(lead_time_days + additional_delay_days)
    arrival_date = pd.Timestamp(start).normalize() + pd.Timedelta(days=lead_days) if order_qty > 0 else pd.NaT
    order_received = bool(order_qty > 0 and lead_days <= horizon)
    item_name = str(stock.item.iloc[0])
    rows: list[dict[str, object]] = []

    for day_index, day_demand in enumerate(demand):
        date = pd.Timestamp(start).normalize() + pd.Timedelta(days=day_index + 1)
        incoming = 0.0
        arrives_today = bool(order_qty > 0 and date == arrival_date and date <= pd.Timestamp(start).normalize() + pd.Timedelta(days=horizon))
        if arrives_today:
            incoming = float(order_qty)
            stock.loc[len(stock)] = [
                f"SIM-ORDER-{date:%Y%m%d}", item_name, incoming, date,
                date + pd.Timedelta(days=order_shelf_life_days), 0.0,
            ]

        stock, served, shortage = allocate_fefo(stock, float(day_demand), date)
        expired_mask = stock.expiry_date < date
        waste = float(stock.loc[expired_mask, "quantity"].clip(lower=0).sum())
        stock = stock.loc[~expired_mask].copy().reset_index(drop=True)
        closing = float(stock.quantity.clip(lower=0).sum())
        usable_closing = inventory_summary(stock, date)["usable"]
        rows.append({
            "date": date,
            "demand": float(day_demand),
            "served": served,
            "shortage": shortage,
            "incoming": incoming,
            "expired_waste": waste,
            "closing_stock": closing,
            "usable_closing_stock": usable_closing,
        })

    result = pd.DataFrame(rows)
    result.attrs.update({
        "order_qty": float(order_qty),
        "order_arrival_date": arrival_date,
        "order_received": order_received,
        "delivery_after_horizon": bool(order_qty > 0 and lead_days > horizon),
        "horizon_end": pd.Timestamp(start).normalize() + pd.Timedelta(days=horizon),
    })
    return result


def summarize_simulation(result: pd.DataFrame) -> dict[str, float | int | bool | pd.Timestamp | None]:
    """Aggregate shortage, fill rate, expiry waste, and receipt evidence."""
    if result.empty:
        raise ValueError("Cannot summarize an empty simulation.")
    demand = float(result.demand.sum())
    served = float(result.served.sum())
    shortages = result[result.shortage > 1e-9]
    received = bool(result.attrs.get("order_received", False))
    arrival = result.attrs.get("order_arrival_date", pd.NaT)
    return {
        "total_demand": demand,
        "served": served,
        "shortage": float(result.shortage.sum()),
        "fill_rate_pct": 100.0 * served / demand if demand > 0 else 100.0,
        "expired_waste": float(result.expired_waste.sum()),
        "ending_stock": float(result.closing_stock.iloc[-1]),
        "first_shortage_date": pd.Timestamp(shortages.date.iloc[0]) if not shortages.empty else None,
        "order_qty": float(result.attrs.get("order_qty", 0.0)),
        "order_received": received,
        "order_arrival_date": arrival,
        "delivery_after_horizon": bool(result.attrs.get("delivery_after_horizon", False)),
    }
