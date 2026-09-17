import unittest

import numpy as np

from codes._unified_evaluation import bhs, compute_metrics


class UnifiedMetricsTests(unittest.TestCase):
    def test_identical_signals_pass_all_metrics(self):
        gt = np.array([[50.0, 75.0, 100.0], [60.0, 80.0, 120.0]])
        result = compute_metrics(gt.copy(), gt)
        self.assertEqual(result["waveform"]["mean"], 0.0)
        for key in ("sbp", "dbp", "map"):
            self.assertEqual(result["pressure"][key]["bhs"]["grade"], "A")
            self.assertEqual(result["pressure"][key]["aami"], "PASS")

    def test_signed_error_direction_is_prediction_minus_gt(self):
        gt = np.array([[50.0, 75.0, 100.0], [60.0, 80.0, 120.0]])
        result = compute_metrics(gt + 3.0, gt)
        for key in ("sbp", "dbp", "map"):
            self.assertAlmostEqual(result["pressure"][key]["signed"]["mean"], 3.0)

    def test_bhs_requires_all_thresholds_for_grade(self):
        error = np.concatenate((np.zeros(60), np.full(25, 8.0), np.full(10, 13.0), np.full(5, 20.0)))
        self.assertEqual(bhs(error)["grade"], "A")
        self.assertEqual(bhs(np.full(100, 20.0))["grade"], "D")


if __name__ == "__main__":
    unittest.main()
