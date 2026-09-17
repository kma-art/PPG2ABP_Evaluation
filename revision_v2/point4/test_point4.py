from __future__ import division

import json
import os
import tempfile
import unittest

import numpy as np

from . import cli
from .core import (SEED, bootstrap_mean_ci, case_seed, dataset_statistics,
                   group_means, holm_adjust, mimic_cluster_included,
                   normalize_ppg, select_protocol_rows,
                   selected_window_ranks, validate_protocol_rows)


class StatisticsTests(unittest.TestCase):
    def test_cluster_equal_differs_from_segment_weighted_for_unequal_clusters(self):
        values = np.asarray([0.0, 0.0, 0.0, 10.0])
        clusters = np.asarray(["large", "large", "large", "small"])
        names, means = group_means(values, clusters)
        self.assertEqual(names, ["large", "small"])
        self.assertEqual(float(np.mean(values)), 2.5)
        self.assertEqual(float(np.mean(means)), 5.0)

    def test_bootstrap_seed_is_exactly_reproducible(self):
        values = np.asarray([1.0, 2.0, 7.0, 11.0])
        first = bootstrap_mean_ci(values, seed=SEED, replicates=1000)
        second = bootstrap_mean_ci(values, seed=SEED, replicates=1000)
        self.assertEqual(first, second)

    def test_dataset_statistics_preserves_both_estimands(self):
        clusters = np.asarray(["a", "a", "b"])
        errors = {}
        for key in ("waveform", "sbp_abs", "dbp_abs", "map_abs",
                    "sbp_signed", "dbp_signed", "map_signed"):
            errors[key] = np.asarray([0.0, 0.0, 9.0])
        result = dataset_statistics(errors, clusters, seed=3)
        self.assertEqual(result["metrics"]["waveform"]["segment_weighted"]["mean"], 3.0)
        self.assertEqual(result["metrics"]["waveform"]["cluster_equal"]["mean"], 4.5)
        self.assertEqual(result["n_clusters"], 2)

    def test_ambiguous_mimic_is_excluded_only_from_cluster_analysis(self):
        statuses = ["exact_unique", "ambiguous", "exact_record_only"]
        mask = [mimic_cluster_included(value) for value in statuses]
        self.assertEqual(mask, [True, False, True])
        self.assertEqual(len(statuses), 3)  # full descriptive N remains intact

    def test_holm_step_down_and_original_order(self):
        adjusted = holm_adjust([0.04, 0.001, 0.03, 0.2])
        np.testing.assert_allclose(adjusted, [0.09, 0.004, 0.09, 0.2])


class ProtocolTests(unittest.TestCase):
    def test_selection_excludes_original_subjects_and_repeats(self):
        cases = list(range(1, 205))
        subjects = dict((case, case + 1000) for case in cases)
        subjects[101] = subjects[1]   # old subject: excluded
        subjects[103] = subjects[102]  # repeated new subject: excluded
        subjects.pop(104)              # missing: excluded
        rows = select_protocol_rows(cases, subjects)
        checks = validate_protocol_rows(
            rows, cases[:100], [subjects[c] for c in cases[:100]])
        self.assertTrue(all(checks.values()))
        self.assertEqual(rows[0]["case_id"], 102)
        self.assertEqual(rows[1]["case_id"], 105)

    def test_window_sampling_is_case_derived_and_processing_order_independent(self):
        cases = [444, 22, 908]
        forward = dict((case, selected_window_ranks(307, case)) for case in cases)
        reverse = dict((case, selected_window_ranks(307, case)) for case in reversed(cases))
        self.assertEqual(forward, reverse)
        self.assertTrue(all(len(value) == 100 for value in forward.values()))
        self.assertNotEqual(case_seed(cases[0]), case_seed(cases[1]))

    def test_existing_protocol_is_immutable(self):
        old_public, old_private = cli.PUBLIC_PROTOCOL, cli.PRIVATE_PROTOCOL
        with tempfile.TemporaryDirectory() as directory:
            try:
                cli.PUBLIC_PROTOCOL = os.path.join(directory, "protocol.json")
                cli.PRIVATE_PROTOCOL = os.path.join(directory, "private.json")
                with open(cli.PUBLIC_PROTOCOL, "w") as handle:
                    json.dump({}, handle)
                with self.assertRaises(RuntimeError):
                    cli.protocol_command(None)
            finally:
                cli.PUBLIC_PROTOCOL, cli.PRIVATE_PROTOCOL = old_public, old_private


class NormalizationTests(unittest.TestCase):
    def test_frozen_normalizations(self):
        x = np.asarray([-16.80837555484424, 0.8139579029352338,
                        66.95604975066223, 96.87874678552862])
        a = normalize_ppg(x, "A")
        b = normalize_ppg(x, "B")
        c = normalize_ppg(x, "C")
        self.assertEqual(float(a[0]), 0.0)
        self.assertEqual(float(a[-1]), 1.0)
        self.assertAlmostEqual(float(b[1]), x[1] / 4.001955034213099, places=6)
        self.assertEqual(float(c[0]), np.float32(-0.1))
        self.assertEqual(float(c[-1]), np.float32(1.1))
        self.assertEqual(float(c[1]), 0.0)
        self.assertEqual(float(c[2]), 1.0)

    def test_normalization_does_not_depend_on_holdout_extrema(self):
        ordinary = normalize_ppg(np.asarray([0.0]), "A")[0]
        with_extreme = normalize_ppg(np.asarray([0.0, 1e12]), "A")[0]
        self.assertEqual(float(ordinary), float(with_extreme))


if __name__ == "__main__":
    unittest.main()
