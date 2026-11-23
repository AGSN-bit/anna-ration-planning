import unittest

import numpy as np
import pandas as pd

from anna.inventory import (
    allocate_fefo,
    apply_demand_scenario,
    inventory_summary,
    recommend_order,
    simulate_policy,
    summarize_simulation,
)


def batches_for(start, rows):
    return pd.DataFrame(
        [
            {
                "batch_id": batch_id,
                "item": "Rice",
                "quantity": quantity,
                "received_date": start - pd.Timedelta(days=10),
                "expiry_date": start + pd.Timedelta(days=expiry_days),
            }
            for batch_id, quantity, expiry_days in rows
        ]
    )


class InventoryCalculationTests(unittest.TestCase):
    def test_scenario_demand_is_separate_from_baseline(self):
        baseline = np.array([10.0, 20.0, 30.0])
        adjusted = apply_demand_scenario(baseline, demand_multiplier=1.2, interruption_days=1)
        np.testing.assert_allclose(baseline, [10, 20, 30])
        np.testing.assert_allclose(adjusted, [13.8, 24, 36])

    def test_fefo_skips_expired_and_consumes_earliest_eligible_first(self):
        start = pd.Timestamp("2026-01-01")
        batches = batches_for(start, [("expired", 50, -1), ("late", 20, 30), ("soon", 8, 3)])
        remaining, served, shortage = allocate_fefo(batches, 10, start)
        allocated = remaining.set_index("batch_id").allocated
        self.assertEqual(allocated["expired"], 0)
        self.assertEqual(allocated["soon"], 8)
        self.assertEqual(allocated["late"], 2)
        self.assertEqual(served, 10)
        self.assertEqual(shortage, 0)

    def test_inventory_summary_excludes_expired_from_usable_stock(self):
        start = pd.Timestamp("2026-01-01")
        batches = batches_for(start, [("expired", 5, -1), ("near", 7, 10), ("later", 11, 50)])
        result = inventory_summary(batches, start, alert_days=30)
        self.assertEqual(result, {"on_hand": 23.0, "usable": 18.0, "expired": 5.0, "near_expiry": 7.0})

    def test_recommendation_uses_lead_demand_safety_stock_and_usable_position(self):
        start = pd.Timestamp("2026-01-01")
        batches = batches_for(start, [("old", 5, -1), ("usable", 10, 90)])
        result = recommend_order(
            [4, 5, 6, 7], batches, start, lead_time_days=2,
            horizon_days=4, daily_demand_std=2, service_factor=1.28,
            confirmed_incoming=3,
        )
        safety = 1.28 * 2 * np.sqrt(2)
        self.assertEqual(result["lead_demand"], 9)
        self.assertAlmostEqual(result["reorder_point"], 9 + safety)
        self.assertEqual(result["inventory_position"], 13)
        self.assertAlmostEqual(result["target_stock"], 22 + safety)
        self.assertAlmostEqual(result["order_qty"], 22 + safety - 13)

    def test_order_arrives_on_date_and_only_changes_post_arrival_days(self):
        start = pd.Timestamp("2026-01-01")
        batches = batches_for(start, [("base", 6, 30)])
        demand = [6, 6, 6, 6]
        no_order = simulate_policy(batches, demand, start, 4, lead_time_days=2, additional_delay_days=1)
        with_order = simulate_policy(batches, demand, start, 4, lead_time_days=2, additional_delay_days=1, order_qty=12)
        self.assertEqual(with_order.date.iloc[2], start + pd.Timedelta(days=3))
        self.assertEqual(with_order.incoming.tolist(), [0, 0, 12, 0])
        np.testing.assert_allclose(with_order.shortage.iloc[:2], no_order.shortage.iloc[:2])
        self.assertEqual(summarize_simulation(no_order)["shortage"], 18)
        self.assertEqual(summarize_simulation(with_order)["shortage"], 6)
        self.assertEqual(summarize_simulation(with_order)["fill_rate_pct"], 75)

    def test_order_after_horizon_is_not_received_or_counted_as_relief(self):
        start = pd.Timestamp("2026-01-01")
        batches = batches_for(start, [("empty", 0, 30)])
        result = simulate_policy(
            batches, [5, 5, 5, 5], start, 4, lead_time_days=3,
            additional_delay_days=2, order_qty=20,
        )
        summary = summarize_simulation(result)
        self.assertTrue(summary["delivery_after_horizon"])
        self.assertFalse(summary["order_received"])
        self.assertEqual(result.incoming.sum(), 0)
        self.assertEqual(summary["shortage"], 20)

    def test_expired_batches_are_wasted_not_consumed(self):
        start = pd.Timestamp("2026-01-01")
        batches = batches_for(start, [("expired", 4, -1), ("expires-tomorrow", 10, 1)])
        result = simulate_policy(batches, [5, 5], start, 2, lead_time_days=1)
        self.assertEqual(result.served.tolist(), [5, 0])
        self.assertEqual(result.shortage.tolist(), [0, 5])
        self.assertEqual(result.expired_waste.tolist(), [4, 5])
        self.assertEqual(summarize_simulation(result)["expired_waste"], 9)


if __name__ == "__main__":
    unittest.main()
