import copy
import json
import tempfile
import unittest
from pathlib import Path

from . import cli
from .core import (assert_close, canonical_json_bytes, csv_bytes,
                   find_forbidden_keys, markdown_bytes, pretty_json_bytes,
                   require_keys, sha256_file)


class Point6CoreTests(unittest.TestCase):
    def test_canonical_json_is_order_independent(self):
        self.assertEqual(canonical_json_bytes({"b": 2, "a": 1}),
                         canonical_json_bytes({"a": 1, "b": 2}))

    def test_pretty_json_has_stable_trailing_newline(self):
        payload = pretty_json_bytes({"value": 1})
        self.assertTrue(payload.endswith(b"\n"))
        self.assertEqual(json.loads(payload), {"value": 1})

    def test_csv_and_markdown_follow_declared_field_order(self):
        rows = [{"b": 2, "a": 1}]
        self.assertEqual(csv_bytes(rows, ["a", "b"]), b"a,b\n1,2\n")
        self.assertEqual(markdown_bytes(rows, ["a", "b"]),
                         b"| a | b |\n|---|---|\n| 1 | 2 |\n")

    def test_identifier_key_filter_is_recursive(self):
        self.assertEqual(find_forbidden_keys({"safe": [{"case_id": 1}]}),
                         ["root.safe[0].case_id"])
        self.assertEqual(find_forbidden_keys({"dataset": "mimic"}), [])

    def test_required_keys_and_exact_tolerance(self):
        require_keys({"a": 1}, ("a",), "fixture")
        assert_close(1.0, 1.0 + 1e-13, "fixture")
        with self.assertRaises(AssertionError):
            require_keys({}, ("a",), "fixture")
        with self.assertRaises(AssertionError):
            assert_close(1.0, 1.1, "fixture")

    def test_internal_digests_detect_data_weight_and_prediction_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                "source_data": Path(directory) / "source.p",
                "model_weights": Path(directory) / "weights.pth",
                "prediction_cache": Path(directory) / "prediction.npy",
            }
            for role, path in paths.items():
                path.write_bytes((role + ":original").encode("utf-8"))
            expected = {role: sha256_file(path) for role, path in paths.items()}
            for changed_role, path in paths.items():
                original = path.read_bytes()
                path.write_bytes(original + b":changed")
                changed = {role for role, candidate in paths.items()
                           if sha256_file(candidate) != expected[role]}
                self.assertEqual(changed, {changed_role})
                path.write_bytes(original)


class Point6RepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cli._validate_nested_integrity()
        cls.sources = cli._load_sources()
        cli._cross_validate(*cls.sources)
        cls.bundle = cli.build_bundle()

    def test_source_precedence_is_explicit(self):
        self.assertTrue(self.bundle["source_precedence"][0].startswith("point5:"))
        self.assertTrue(self.bundle["source_precedence"][-1].startswith("point2:"))
        roles = {row["name"]: row["role"] for row in self.bundle["sources"]}
        self.assertEqual(roles["point2_results"], "legacy scale audit only")

    def test_bundle_is_deterministic_and_identifier_free(self):
        self.assertEqual(pretty_json_bytes(self.bundle),
                         pretty_json_bytes(cli.build_bundle()))
        self.assertEqual(find_forbidden_keys(self.bundle), [])

    def test_public_bundle_and_tables_have_no_sha_fields(self):
        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield str(key).lower()
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        self.assertEqual(self.bundle["schema_version"], 2)
        self.assertFalse(any("sha" in key for key in keys(self.bundle)))
        for table in self.bundle["tables"].values():
            self.assertFalse(any("sha" in field.lower() for field in table["fields"]))

    def test_local_integrity_covers_inputs_and_prediction_caches(self):
        integrity = json.loads(cli.INTEGRITY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(integrity["schema_version"], 2)
        self.assertEqual(integrity["inputs"], cli._input_hashes())
        self.assertEqual(integrity["prediction_caches"], cli._prediction_hashes())
        self.assertTrue(integrity["prediction_caches"])

    def test_cross_stage_count_mismatch_fails(self):
        point2, provenance, point4, point5 = self.sources
        changed = copy.deepcopy(point5)
        changed["datasets"]["mimic"]["scenarios"]["raw"]["coverage"]["segments_total"] += 1
        with self.assertRaises(AssertionError):
            cli._cross_validate(point2, provenance, point4, changed)

    def test_primary_and_secondary_figure_statuses(self):
        rows = self.bundle["tables"]["figure_index"]["rows"]
        ready = [row for row in rows if row["status"] == "ready_point5"]
        legacy = [row for row in rows if row["status"] == "legacy_review_required"]
        self.assertEqual(len(ready), 6)
        self.assertEqual(len(legacy), 2)
        self.assertTrue(all(min(row["dpi_x"], row["dpi_y"]) >= 299 for row in ready))

    def test_all_derived_tables_have_rows_and_unique_fields(self):
        for name, table in self.bundle["tables"].items():
            self.assertTrue(table["rows"], name)
            self.assertEqual(len(table["fields"]), len(set(table["fields"])), name)
            self.assertTrue(all(set(row) <= set(table["fields"]) for row in table["rows"]), name)


if __name__ == "__main__":
    unittest.main()
