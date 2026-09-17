import os
import tempfile
import unittest

import numpy as np

import segment_provenance as provenance


class HashMatchingTests(unittest.TestCase):
    def test_hash_is_exact_and_channel_ordered(self):
        ppg = np.array([1.0, 2.0], dtype=np.float64)
        abp = np.array([80.0, 120.0], dtype=np.float64)
        self.assertEqual(provenance.signal_hash(ppg, abp),
                         provenance.signal_hash(ppg.copy(), abp.copy()))
        self.assertNotEqual(provenance.signal_hash(ppg, abp),
                            provenance.signal_hash(abp, ppg))
        self.assertNotEqual(provenance.signal_hash(ppg, abp),
                            provenance.signal_hash(ppg.astype(np.float32), abp.astype(np.float32)))

    def test_match_classification_handles_duplicates(self):
        unique = [{"part": 1, "record_idx": 2, "episode_start": 0}]
        same_record = unique + [{"part": 1, "record_idx": 2, "episode_start": 625}]
        different_records = unique + [{"part": 1, "record_idx": 3, "episode_start": 0}]
        self.assertEqual(provenance.classify_matches([]), "unmatched")
        self.assertEqual(provenance.classify_matches(unique), "exact_unique")
        self.assertEqual(provenance.classify_matches(unique + unique), "exact_unique")
        self.assertEqual(provenance.classify_matches(same_record), "exact_record_only")
        self.assertEqual(provenance.classify_matches(different_records), "ambiguous")


class FilteringAndSplitTests(unittest.TestCase):
    def test_abp_ppg_filter_reasons_can_overlap(self):
        flat = np.ones(1024, dtype=np.float64)
        bad_abp = np.linspace(-1, 301, 1024, dtype=np.float64)
        self.assertEqual(provenance.filter_abp_ppg(flat, bad_abp),
                         ["flat_ppg", "abp_negative", "abp_outside_20_300"])
        good_ppg = np.sin(np.linspace(0, 10, 1024))
        good_abp = np.linspace(60, 120, 1024)
        self.assertEqual(provenance.filter_abp_ppg(good_ppg, good_abp), [])

    def test_fold_9_roles(self):
        self.assertEqual(provenance.split_role(0), "train")
        self.assertEqual(provenance.split_role(79999), "train")
        self.assertEqual(provenance.split_role(80000), "validation")
        self.assertEqual(provenance.split_role(89999), "validation")
        self.assertEqual(provenance.split_role(90000), "train")
        self.assertEqual(provenance.split_role(100000), "test")

    def test_vital_window_filter_and_rank(self):
        ppg = np.sin(np.linspace(0, 100, 2049))
        abp = 90 + 20 * np.sin(np.linspace(0, 100, 2049))
        windows = list(provenance.vital_windows(ppg, abp))
        # The original range excludes a window ending exactly at n_samples.
        self.assertEqual([(w[0], w[1]) for w in windows], [(0, 0), (1, 512), (2, 1024)])

    def test_csv_count(self):
        descriptor, path = tempfile.mkstemp(suffix=".csv")
        os.close(descriptor)
        try:
            provenance.write_csv(path, ["a"], [{"a": 1}, {"a": 2}])
            self.assertEqual(provenance.csv_row_count(path), 2)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
