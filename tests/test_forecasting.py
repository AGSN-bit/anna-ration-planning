import unittest

import numpy as np
import pandas as pd

from anna.forecasting import evaluate_holdout, forecast


class ForecastingTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.Series(pd.date_range("2025-01-01", periods=120, freq="D"))
        self.values = pd.Series(10.0 + np.arange(120, dtype=float) * 0.1)

    def test_moving_average_uses_recent_28_day_mean(self):
        expected = self.values.tail(28).mean()
        result = forecast(self.values, self.dates, 5, "Moving average", 7)
        np.testing.assert_allclose(result, np.full(5, expected))

    def test_random_forest_returns_nonnegative_forecast_for_horizon(self):
        result = forecast(self.values, self.dates, 7, "Random Forest", 7)
        self.assertEqual(len(result), 7)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0).all())

    def test_holdout_metrics_use_chronological_training_prefix(self):
        holdout = 14
        train = self.values.iloc[:-holdout].reset_index(drop=True)
        train_dates = self.dates.iloc[:-holdout].reset_index(drop=True)
        actual = self.values.iloc[-holdout:].to_numpy()
        predicted = forecast(train, train_dates, holdout, "Moving average", 7)
        expected_mae = float(np.abs(actual - predicted).mean())
        metrics = evaluate_holdout(self.values, self.dates, "Moving average", 7, holdout=holdout)
        self.assertAlmostEqual(metrics["MAE"], expected_mae)
        self.assertGreaterEqual(metrics["RMSE"], metrics["MAE"])
        self.assertTrue(np.isfinite(metrics["WAPE"]))

    def test_forecast_rejects_misaligned_or_unsorted_dates(self):
        with self.assertRaises(ValueError):
            forecast(self.values, self.dates.iloc[:-1], 7, "Moving average", 7)
        with self.assertRaises(ValueError):
            forecast(self.values, self.dates.iloc[::-1].reset_index(drop=True), 7, "Moving average", 7)


if __name__ == "__main__":
    unittest.main()
