from __future__ import division, print_function

import unittest

import numpy as np

from .core import (bland_altman_statistics, bootstrap_mean_ci,
                   calibration_bins, metric_statistics, scenario_masks,
                   scenario_statistics, weighted_quantile)
from .cli import (PREDICTION_REFERENCE_ZOOM, _error_vectors,
                  _outside_plot_limits)


def minimal_columns(waveform, clusters=None, included=None, outside=None,
                    gt_min=None, gt_max=None):
    waveform = np.asarray(waveform, dtype=np.float64)
    count = len(waveform)
    if clusters is None:
        clusters = ["c{0}".format(index) for index in range(count)]
    if included is None:
        included = np.ones(count, dtype=np.int64)
    if outside is None:
        outside = np.zeros(count, dtype=np.int64)
    if gt_min is None:
        gt_min = np.full(count, 60.0)
    if gt_max is None:
        gt_max = np.full(count, 180.0)
    result = {"cluster_id": np.asarray(clusters),
              "cluster_included": np.asarray(included),
              "pred_outside_20_300": np.asarray(outside),
              "gt_min": np.asarray(gt_min), "gt_max": np.asarray(gt_max)}
    for prefix in ("raw_", "clip_"):
        for metric in ("waveform", "sbp_abs", "dbp_abs", "map_abs",
                       "sbp_signed", "dbp_signed", "map_signed"):
            result[prefix + metric] = waveform.copy()
    return result


class Point5Tests(unittest.TestCase):
    def test_prediction_reference_zoom_limits_are_pressure_specific(self):
        self.assertEqual(PREDICTION_REFERENCE_ZOOM["sbp"], (60.0, 220.0))
        self.assertEqual(PREDICTION_REFERENCE_ZOOM["dbp"], (20.0, 140.0))
        actual = _outside_plot_limits([59, 60, 140, 220],
                                      [100, 60, 141, 221], (60, 220))
        np.testing.assert_array_equal(actual, [True, False, False, True])

    def test_unequal_cluster_sizes_have_distinct_estimands(self):
        stats = metric_statistics(np.asarray([0.0, 0.0, 9.0]),
                                  np.asarray(["a", "a", "b"]),
                                  replicates=100)
        self.assertEqual(stats["segment_weighted"]["mean"], 3.0)
        self.assertEqual(stats["cluster_equal"]["mean"], 4.5)

    def test_weighted_quantile_uses_weights(self):
        actual = weighted_quantile([0, 10, 20], [.8, .1, .1], [.5, .9, .95])
        np.testing.assert_array_equal(actual, [0, 10, 20])

    def test_bootstrap_seed_is_reproducible(self):
        first = bootstrap_mean_ci([1, 2, 8, 10], seed=20260908, replicates=37)
        second = bootstrap_mean_ci([1, 2, 8, 10], seed=20260908, replicates=37)
        other = bootstrap_mean_ci([1, 2, 8, 10], seed=7, replicates=37)
        self.assertEqual(first, second)
        self.assertEqual(first, [1.475, 8.600000000000001])
        self.assertNotEqual(first, other)

    def test_ambiguous_mimic_stays_segment_weighted_only(self):
        columns = minimal_columns([1, 3, 100], ["a", "b", ""], [1, 1, 0])
        stats = scenario_statistics(columns, "raw", np.ones(3, dtype=bool),
                                    replicates=100)
        self.assertAlmostEqual(stats["metrics"]["waveform"]["segment_weighted"]["mean"],
                               104.0 / 3.0)
        self.assertEqual(stats["metrics"]["waveform"]["cluster_equal"]["mean"], 2.0)
        self.assertEqual(stats["coverage"]["cluster_excluded_segments"], 1)

    def test_pointwise_clipping_retains_n_and_bounds_prediction(self):
        prediction = np.asarray([[0.0, 100.0, 400.0]])
        truth = np.asarray([[20.0, 100.0, 300.0]])
        clipped = np.clip(prediction, 20.0, 300.0)
        self.assertEqual(len(clipped), len(prediction))
        self.assertGreaterEqual(float(np.min(clipped)), 20.0)
        self.assertLessEqual(float(np.max(clipped)), 300.0)
        self.assertEqual(_error_vectors(clipped, truth)["waveform"][0], 0.0)

    def test_reject_mask_does_not_use_ground_truth(self):
        first = minimal_columns([1, 2, 3], outside=[0, 1, 0],
                                gt_min=[60, 60, 60], gt_max=[180, 180, 180])
        second = minimal_columns([900, 800, 700], outside=[0, 1, 0],
                                 gt_min=[0, 500, -10], gt_max=[900, 800, 700])
        np.testing.assert_array_equal(scenario_masks(first)[0]["reject"],
                                      scenario_masks(second)[0]["reject"])

    def test_oracle_top5_is_truth_error_dependent(self):
        values = np.arange(20, dtype=np.float64)
        first = minimal_columns(values)
        second = minimal_columns(values[::-1])
        masks_first = scenario_masks(first)[0]["oracle_top5"]
        masks_second = scenario_masks(second)[0]["oracle_top5"]
        self.assertFalse(np.array_equal(masks_first, masks_second))
        self.assertEqual(int(np.sum(masks_first)), 19)

    def test_gt_subgroup_rule(self):
        columns = minimal_columns([1, 2, 3, 4], gt_min=[50, 49, 50, 60],
                                  gt_max=[200, 200, 201, 190])
        np.testing.assert_array_equal(scenario_masks(columns)[0]["gt_50_200"],
                                      [True, False, False, True])

    def test_calibration_bins_require_ten_clusters(self):
        reference = np.asarray([31.0] * 10 + [51.0] * 9)
        prediction = reference + 2
        clusters = np.asarray(["a{0}".format(i) for i in range(10)] +
                              ["b{0}".format(i) for i in range(9)])
        bins = calibration_bins(reference, prediction, clusters)
        self.assertTrue(bins[0]["shown"])
        self.assertFalse(bins[1]["shown"])
        self.assertEqual(bins[0]["prediction_mean"], 33.0)

    def test_bland_altman_equal_cluster_mixture(self):
        result = bland_altman_statistics([0, 0, 10], ["a", "a", "b"],
                                         seed=20260908, replicates=200)
        self.assertEqual(result["bias"], 5.0)
        self.assertEqual(result["nonparametric_limits"], [0.0, 10.0])
        self.assertAlmostEqual(result["traditional_limits"][0], -4.8)
        self.assertAlmostEqual(result["traditional_limits"][1], 14.8)


if __name__ == "__main__":
    unittest.main()
