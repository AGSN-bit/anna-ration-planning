"""Small, leakage-aware forecasting helpers used by the demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _features(values: list[float], next_date: pd.Timestamp) -> list[float]:
    def lag(k: int) -> float:
        return float(values[-k]) if len(values) >= k else float(np.mean(values))

    rolling_7 = float(np.mean(values[-7:]))
    rolling_28 = float(np.mean(values[-28:]))
    dow = next_date.day_of_week
    doy = next_date.day_of_year
    return [lag(1), lag(7), lag(14), lag(28), rolling_7, rolling_28,
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]


def _training_rows(series: pd.Series, dates: pd.Series, first_target: int) -> tuple[np.ndarray, np.ndarray]:
    values = [float(x) for x in series]
    X, y = [], []
    for i in range(max(28, first_target), len(values)):
        X.append(_features(values[:i], pd.Timestamp(dates.iloc[i])))
        y.append(values[i])
    return np.asarray(X), np.asarray(y)


def forecast(series: pd.Series, dates: pd.Series, horizon: int, model: str, seed: int) -> np.ndarray:
    """Forecast daily demand recursively using moving average or Random Forest."""
    values = [float(v) for v in series]
    if not values:
        raise ValueError("At least one historical observation is required.")
    if len(values) != len(dates):
        raise ValueError("Demand history and dates must have the same length.")
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one day.")
    if model not in {"Moving average", "Random Forest"}:
        raise ValueError(f"Unsupported forecast method: {model}")
    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    if date_index.has_duplicates or not date_index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted in ascending order.")
    if model == "Moving average":
        level = float(np.mean(values[-28:]))
        return np.full(horizon, level, dtype=float)

    X, y = _training_rows(series.reset_index(drop=True), dates.reset_index(drop=True), 28)
    if len(y) < 30:
        return np.full(horizon, float(np.mean(values[-28:])), dtype=float)
    estimator = RandomForestRegressor(
        n_estimators=160, min_samples_leaf=3, max_features=0.9,
        random_state=seed, n_jobs=1,
    )
    estimator.fit(X, y)
    next_dates = pd.date_range(pd.Timestamp(dates.iloc[-1]) + pd.Timedelta(days=1), periods=horizon, freq="D")
    predictions = []
    for next_date in next_dates:
        prediction = max(0.0, float(estimator.predict([_features(values, next_date)])[0]))
        predictions.append(prediction)
        values.append(prediction)
    return np.asarray(predictions)


def evaluate_holdout(series: pd.Series, dates: pd.Series, model: str, seed: int, holdout: int = 28) -> dict[str, float]:
    """Evaluate a single chronological holdout; future values never enter training."""
    if len(series) != len(dates):
        raise ValueError("Demand history and dates must have the same length.")
    if holdout < 1:
        raise ValueError("Holdout must be at least one day.")
    if len(series) <= holdout + 30:
        return {"MAE": float("nan"), "RMSE": float("nan"), "WAPE": float("nan")}
    train = series.iloc[:-holdout].reset_index(drop=True)
    train_dates = dates.iloc[:-holdout].reset_index(drop=True)
    actual = series.iloc[-holdout:].to_numpy(dtype=float)
    predicted = forecast(train, train_dates, holdout, model, seed)
    errors = actual - predicted
    denominator = float(np.abs(actual).sum())
    return {
        "MAE": float(np.abs(errors).mean()),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "WAPE": float(np.abs(errors).sum() / denominator * 100) if denominator else float("nan"),
    }
