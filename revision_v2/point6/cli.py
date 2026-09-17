"""Build and verify the canonical article-material bundle for revision point 6.

This orchestrator never performs inference or data acquisition.  ``build``
uses only frozen aggregate outputs; ``verify`` validates them quickly;
``verify --full`` runs the existing point 2--5 verification commands after a
no-inference preflight.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .core import (assert_close, canonical_json_bytes, csv_bytes,
                   find_forbidden_keys, markdown_bytes, pretty_json_bytes,
                   read_json, require_keys, sha256_file,
                   write_if_changed)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODES = ROOT / "codes"
DATA = CODES / "data"
POINT4_ROOT = DATA / "revision_v2" / "point4" / "v1"
POINT5_ROOT = DATA / "revision_v2" / "point5" / "v1"
PRIVATE_ROOT = DATA / "revision_v2" / "point6" / "v2"
FULL_VERIFY_PATH = PRIVATE_ROOT / "full_verify.json"
TABLE_ROOT = CODES / "tables" / "revision_v2" / "point6"
BUNDLE_PATH = HERE / "article_bundle_v2.json"
INTEGRITY_PATH = PRIVATE_ROOT / "integrity.json"
NOTEBOOK_PATH = CODES / "_article_revision_v2.ipynb"
REPORT_PATH = ROOT / ".context" / "REVISION_POINT6.md"

POINT2_RESULTS = DATA / "unified_evaluation_cache" / "metrics.json"
POINT2_REPORT = ROOT / ".context" / "EVALUATION_RESULTS.md"
PROVENANCE_SUMMARY = DATA / "provenance" / "summary.json"
PROVENANCE_REPORT = ROOT / ".context" / "SEGMENT_PROVENANCE.md"
POINT4_RESULTS = POINT4_ROOT / "results.json"
POINT4_INTEGRITY = POINT4_ROOT / "integrity.json"
POINT5_RESULTS = POINT5_ROOT / "results.json"
POINT5_INTEGRITY = POINT5_ROOT / "integrity.json"

DATASET_ORDER = ("mimic", "abp_ppg", "vitaldb_old", "holdout_A",
                 "holdout_B", "holdout_C")
MAIN_DATASETS = ("mimic", "abp_ppg", "vitaldb_old", "holdout_A")
SCENARIO_ORDER = ("raw", "clip", "reject", "oracle_top5", "gt_50_200")
ABS_METRICS = ("waveform", "sbp_abs", "dbp_abs", "map_abs")
SIGNED_METRICS = ("sbp_signed", "dbp_signed", "map_signed")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def _input_paths() -> dict[str, Path]:
    return {
        "point2_results": POINT2_RESULTS,
        "provenance_summary": PROVENANCE_SUMMARY,
        "point4_results": POINT4_RESULTS,
        "point4_integrity": POINT4_INTEGRITY,
        "point5_results": POINT5_RESULTS,
        "point5_integrity": POINT5_INTEGRITY,
    }


def _input_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in _input_paths().items()}


def _validate_nested_integrity() -> None:
    from revision_v2.point4.cli import _artifact_hashes as point4_hashes
    from revision_v2.point5.cli import _artifact_hashes as point5_hashes

    if read_json(POINT4_INTEGRITY)["artifacts"] != point4_hashes(True):
        raise AssertionError("point4 integrity map differs")
    if read_json(POINT5_INTEGRITY)["artifacts"] != point5_hashes(True):
        raise AssertionError("point5 integrity map differs")


def _validate_provenance_summary(summary: dict[str, Any]) -> None:
    require_keys(summary, ("schema_version", "mimic", "abp_ppg", "vitaldb", "report"),
                 "provenance summary")
    if sha256_file(ROOT / summary["report"]["path"]) != summary["report"]["sha256"]:
        raise AssertionError("provenance report hash differs")
    specifications = (
        (summary["mimic"]["manifest"], summary["mimic"]["manifest_rows"],
         summary["mimic"]["manifest_sha256"]),
        (summary["abp_ppg"]["manifest"], summary["abp_ppg"]["manifest_rows"],
         summary["abp_ppg"]["manifest_sha256"]),
        (summary["vitaldb"]["segments_manifest"],
         summary["vitaldb"]["segments_manifest_rows"],
         summary["vitaldb"]["segments_manifest_sha256"]),
        (summary["vitaldb"]["cases_manifest"],
         summary["vitaldb"]["cases_manifest_rows"],
         summary["vitaldb"]["cases_manifest_sha256"]),
    )
    for relative, expected_rows, expected_hash in specifications:
        path = ROOT / relative
        if _csv_rows(path) != expected_rows or sha256_file(path) != expected_hash:
            raise AssertionError("provenance manifest differs: " + relative)


def _load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    point2 = read_json(POINT2_RESULTS)
    provenance = read_json(PROVENANCE_SUMMARY)
    point4 = read_json(POINT4_RESULTS)
    point5 = read_json(POINT5_RESULTS)
    require_keys(point2, ("datasets", "meta", "publication_reproduction_x200"), "point2")
    require_keys(point4, ("datasets", "comparisons", "normalization_role"), "point4")
    require_keys(point5, ("datasets", "old_text_audit", "raw_point4_exact_match"), "point5")
    _validate_provenance_summary(provenance)
    return point2, provenance, point4, point5


def _point2_metric(dataset: dict[str, Any], metric: str, signed: bool = False) -> dict[str, Any]:
    if metric == "waveform":
        return dataset["waveform"]
    pressure = metric[:3]
    return dataset["pressure"][pressure]["signed" if signed else "absolute"]


def _cross_validate(point2: dict[str, Any], provenance: dict[str, Any],
                    point4: dict[str, Any], point5: dict[str, Any]) -> None:
    if point4["seed"] != 20260908 or point5["seed"] != 20260908:
        raise AssertionError("revision seed changed")
    if point4["bootstrap_replicates"] != 10000 or point5["bootstrap_replicates"] != 10000:
        raise AssertionError("bootstrap replicate count changed")
    if point5["primary_scenario"] != "raw" or not all(point5["raw_point4_exact_match"].values()):
        raise AssertionError("point5 no longer declares exact raw equivalence")

    for dataset in DATASET_ORDER:
        p4 = point4["datasets"][dataset]
        raw = point5["datasets"][dataset]["scenarios"]["raw"]
        expected_point4_segments = (raw["coverage"]["segments_total"] -
                                    raw["coverage"]["cluster_excluded_segments"])
        if p4["n_segments"] != expected_point4_segments:
            raise AssertionError("segment count differs for " + dataset)
        if p4["n_clusters"] != raw["coverage"]["clusters_total"]:
            raise AssertionError("cluster count differs for " + dataset)
        for metric in ABS_METRICS + SIGNED_METRICS:
            for estimand in ("segment_weighted", "cluster_equal"):
                current = raw["metrics"][metric][estimand]
                previous = p4["metrics"][metric][estimand]
                if dataset == "mimic" and estimand == "segment_weighted":
                    previous = p4["full_segment_weighted"][metric]
                assert_close(current["mean"], previous["mean"],
                             "point4/5 {0}/{1}/{2} mean".format(dataset, metric, estimand))
                assert_close(current["sd"], previous["sd"],
                             "point4/5 {0}/{1}/{2} sd".format(dataset, metric, estimand))

    for p2_name, p5_name in (("mimic", "mimic"), ("abp_ppg", "abp_ppg"),
                             ("vitaldb", "vitaldb_old")):
        p2_dataset = point2["datasets"][p2_name]
        raw = point5["datasets"][p5_name]["scenarios"]["raw"]
        if p2_dataset["n"] != raw["coverage"]["segments_total"]:
            raise AssertionError("point2/5 N differs for " + p2_name)
        for metric in ABS_METRICS:
            expected = _point2_metric(p2_dataset, metric)
            actual = raw["metrics"][metric]["segment_weighted"]
            assert_close(actual["mean"], expected["mean"],
                         "point2/5 {0}/{1} mean".format(p2_name, metric))
            assert_close(actual["sd"], expected["sd"],
                         "point2/5 {0}/{1} sd".format(p2_name, metric))
        for metric in SIGNED_METRICS:
            expected = _point2_metric(p2_dataset, metric, signed=True)
            actual = raw["metrics"][metric]["segment_weighted"]
            assert_close(actual["mean"], expected["mean"],
                         "point2/5 {0}/{1} mean".format(p2_name, metric))

    expected_counts = {
        "mimic": (provenance["mimic"]["corrected_test_segments"],
                  provenance["mimic"]["unique_records_by_role"]["test"]),
        "abp_ppg": (provenance["abp_ppg"]["accepted_segments"],
                    provenance["abp_ppg"]["segment_subjects"]),
        "vitaldb_old": (provenance["vitaldb"]["saved_segments"],
                        provenance["vitaldb"]["represented_subjects"]),
    }
    for dataset, (segments, clusters) in expected_counts.items():
        raw = point5["datasets"][dataset]["scenarios"]["raw"]["coverage"]
        if raw["segments_total"] != segments or raw["clusters_total"] != clusters:
            raise AssertionError("provenance count differs for " + dataset)


def _source_map() -> list[dict[str, Any]]:
    roles = {
        "point2_results": "legacy scale audit only",
        "provenance_summary": "dataset composition and provenance",
        "point4_results": "holdout comparisons and normalization roles",
        "point5_results": "authoritative final metrics and diagnostics",
    }
    return [{"name": name, "path": _relative(_input_paths()[name]),
             "role": role}
            for name, role in roles.items()]


def _composition_rows(provenance: dict[str, Any], point4: dict[str, Any],
                      point5: dict[str, Any]) -> list[dict[str, Any]]:
    acquisition = read_json(POINT4_ROOT / "acquisition" / "manifest.json")
    return [
        {"dataset": "mimic", "role": "main", "input_entities": provenance["mimic"]["hdf5_rows"],
         "loaded_entities": "", "represented_clusters": point4["datasets"]["mimic"]["n_clusters"],
         "evaluated_segments": provenance["mimic"]["corrected_test_segments"],
         "initial_excluded_segments": 0, "cluster_excluded_segments": 92,
         "notes": "ambiguous provenance retained only in full segment-weighted descriptions"},
        {"dataset": "abp_ppg", "role": "main", "input_entities": provenance["abp_ppg"]["source_segments"],
         "loaded_entities": provenance["abp_ppg"]["files"],
         "represented_clusters": provenance["abp_ppg"]["segment_subjects"],
         "evaluated_segments": provenance["abp_ppg"]["accepted_segments"],
         "initial_excluded_segments": provenance["abp_ppg"]["excluded_segments"],
         "cluster_excluded_segments": 0, "notes": "patient overlap with MIMIC cannot be excluded"},
        {"dataset": "vitaldb_old", "role": "main", "input_entities": provenance["vitaldb"]["requested_cases"],
         "loaded_entities": provenance["vitaldb"]["loaded_cases"],
         "represented_clusters": provenance["vitaldb"]["represented_subjects"],
         "evaluated_segments": provenance["vitaldb"]["saved_segments"],
         "initial_excluded_segments": 0, "cluster_excluded_segments": 0,
         "notes": "original VitalDB normalization A"},
    ] + [
        {"dataset": dataset, "role": ("confirmatory independent holdout" if dataset == "holdout_A"
                                       else "secondary normalization ablation"),
         "input_entities": acquisition["requested_cases"],
         "loaded_entities": acquisition["loaded_cases"],
         "represented_clusters": acquisition["represented_subjects"],
         "evaluated_segments": acquisition["segments"], "initial_excluded_segments": 0,
         "cluster_excluded_segments": 0,
         "notes": "same subject-disjoint holdout signals; normalization " + dataset[-1]}
        for dataset in ("holdout_A", "holdout_B", "holdout_C")
    ]


def _main_metric_rows(point5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        raw = point5["datasets"][dataset]["scenarios"]["raw"]
        for metric in ABS_METRICS:
            for estimand in ("segment_weighted", "cluster_equal"):
                value = raw["metrics"][metric][estimand]
                rows.append({"dataset": dataset, "metric": metric, "estimand": estimand,
                             "unit": "mmHg", "n": value["n"], "mean": value["mean"],
                             "sd": value["sd"],
                             "ci95_lower": value.get("ci95", [None, None])[0],
                             "ci95_upper": value.get("ci95", [None, None])[1]})
    return rows


def _signed_rows(point5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        raw = point5["datasets"][dataset]["scenarios"]["raw"]
        for metric in SIGNED_METRICS:
            for estimand in ("segment_weighted", "cluster_equal"):
                value = raw["metrics"][metric][estimand]
                rows.append({"dataset": dataset, "metric": metric, "estimand": estimand,
                             "unit": "mmHg", "n": value["n"], "mean": value["mean"],
                             "sd": value["sd"],
                             "ci95_lower": value.get("ci95", [None, None])[0],
                             "ci95_upper": value.get("ci95", [None, None])[1]})
    return rows


def _bhs_rows(point5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        raw = point5["datasets"][dataset]["scenarios"]["raw"]
        for pressure in ("sbp", "dbp", "map"):
            for threshold in (5, 10, 15):
                for estimand in ("segment_weighted", "cluster_equal"):
                    value = raw["bhs"][pressure]["le{0}".format(threshold)][estimand]
                    rows.append({"dataset": dataset, "pressure": pressure,
                                 "threshold_mmhg": threshold, "estimand": estimand,
                                 "percent": value["mean"], "sd": value["sd"], "n": value["n"],
                                 "ci95_lower": value.get("ci95", [None, None])[0],
                                 "ci95_upper": value.get("ci95", [None, None])[1]})
    return rows


def _normalization_rows(point4: dict[str, Any], point5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in ("A", "B", "C"):
        dataset = "holdout_" + variant
        raw = point5["datasets"][dataset]["scenarios"]["raw"]
        for metric in ABS_METRICS:
            value = raw["metrics"][metric]["cluster_equal"]
            rows.append({"row_type": "holdout_variant", "variant": variant,
                         "role": point4["normalization_role"][variant], "comparator": "",
                         "metric": metric, "estimate": value["mean"],
                         "ci95_lower": value["ci95"][0], "ci95_upper": value["ci95"][1],
                         "p_holm": None, "cliffs_delta": None})
    for comparison in point4["comparisons"]:
        rows.append({"row_type": "prespecified_comparison", "variant": "A",
                     "role": "confirmatory", "comparator": comparison["right"],
                     "metric": comparison["metric"],
                     "estimate": comparison["difference_cluster_equal_means"],
                     "ci95_lower": comparison["difference_ci95"][0],
                     "ci95_upper": comparison["difference_ci95"][1],
                     "p_holm": comparison["p_holm"],
                     "cliffs_delta": comparison["cliffs_delta"]})
    return rows


def _postprocessing_rows(point5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        for scenario in SCENARIO_ORDER:
            value = point5["datasets"][dataset]["scenarios"][scenario]
            coverage = value["coverage"]
            row = {"dataset": dataset, "scenario": scenario,
                   "uses_ground_truth": value["exclusion"]["uses_ground_truth"],
                   "exclusion_reason": value["exclusion"]["reason"],
                   "segments_retained": coverage["segments_retained"],
                   "segments_total": coverage["segments_total"],
                   "segment_fraction": coverage["segment_fraction"],
                   "clusters_retained": coverage["clusters_retained"],
                   "clusters_total": coverage["clusters_total"],
                   "prediction_outside_events": value["prediction_outside_20_300"]["events"],
                   "prediction_outside_affected_clusters":
                       value["prediction_outside_20_300"]["affected_clusters"],
                   "prediction_points_outside_before_processing":
                       value["prediction_points_outside_20_300_before_processing"]}
            for metric in ABS_METRICS:
                row[metric + "_cluster_equal_mean"] = value["metrics"][metric]["cluster_equal"]["mean"]
            rows.append(row)
    return rows


def _tail_rows(point5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        raw = point5["datasets"][dataset]["scenarios"]["raw"]
        for metric in ABS_METRICS:
            tail = raw["tails"][metric]
            rows.append({"dataset": dataset, "scenario": "raw", "metric": metric,
                         "q1": tail["q1"], "median": tail["median"], "q3": tail["q3"],
                         "p90": tail["p90"], "p95": tail["p95"], "p99": tail["p99"],
                         "p99_5": tail["p99_5"], "maximum": tail["maximum"],
                         "count_gt30": tail["counts_above"]["30"],
                         "count_gt50": tail["counts_above"]["50"],
                         "count_gt100": tail["counts_above"]["100"],
                         "count_gt200": tail["counts_above"]["200"],
                         "count_gt500": tail["counts_above"]["500"]})
    return rows


def _diagnostic_rows(point5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in MAIN_DATASETS:
        diagnostics = point5["datasets"][dataset]["diagnostics"]
        for pressure in ("sbp", "dbp"):
            correlation = diagnostics["correlations"][pressure]["cluster_level_primary"]
            agreement = diagnostics["bland_altman"][pressure]
            row = {"dataset": dataset, "pressure": pressure, "clusters": correlation["clusters"]}
            for metric in ("pearson_r", "spearman_rho", "r2", "calibration_slope",
                           "calibration_intercept"):
                row[metric] = correlation[metric]["estimate"]
                row[metric + "_ci95_lower"] = correlation[metric]["ci95"][0]
                row[metric + "_ci95_upper"] = correlation[metric]["ci95"][1]
            row.update({"ba_bias": agreement["bias"],
                        "ba_bias_ci95_lower": agreement["bias_ci95"][0],
                        "ba_bias_ci95_upper": agreement["bias_ci95"][1],
                        "ba_traditional_lower": agreement["traditional_limits"][0],
                        "ba_traditional_upper": agreement["traditional_limits"][1],
                        "ba_nonparametric_lower": agreement["nonparametric_limits"][0],
                        "ba_nonparametric_upper": agreement["nonparametric_limits"][1],
                        "ba_nonparametric_lower_ci95_lower": agreement["nonparametric_lower_ci95"][0],
                        "ba_nonparametric_lower_ci95_upper": agreement["nonparametric_lower_ci95"][1],
                        "ba_nonparametric_upper_ci95_lower": agreement["nonparametric_upper_ci95"][0],
                        "ba_nonparametric_upper_ci95_upper": agreement["nonparametric_upper_ci95"][1]})
            rows.append(row)
    return rows


def _calibration_rows(point5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in MAIN_DATASETS:
        for pressure in ("sbp", "dbp"):
            for value in point5["datasets"][dataset]["diagnostics"]["calibration"][pressure]:
                rows.append({"dataset": dataset, "pressure": pressure, **value})
    return rows


def _legacy_rows(point2: dict[str, Any], point5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric, value in point2["publication_reproduction_x200"].items():
        rows.append({"audit_type": "publication_x200_reproduction", "metric": metric,
                     "subset": "MIMIC", "n": point2["datasets"]["mimic"]["n"],
                     "mean": None, "sd": None, "paper": value["paper"],
                     "reproduced": value["reproduced"], "deviation": value["deviation"]})
    audit = point5["old_text_audit"]
    for subset in ("top5_retained", "gt_50_200", "intersection_audit_only"):
        value = audit[subset]
        rows.append({"audit_type": "mixed_outlier_value", "metric": "waveform",
                     "subset": subset, "n": value["n"], "mean": value["mean"],
                     "sd": value["sd"], "paper": None, "reproduced": None,
                     "deviation": None})
    return rows


def _figure_rows() -> list[dict[str, Any]]:
    from PIL import Image

    descriptions = {
        "error_distributions": "raw error distributions",
        "prediction_reference_full": "prediction-reference full range",
        "prediction_reference_zoom": "prediction-reference pressure-specific zoom",
        "calibration": "fixed-bin cluster-equal calibration",
        "bland_altman_full": "Bland-Altman full range",
        "bland_altman_zoom": "Bland-Altman difference zoom",
    }
    rows = []
    for name, description in descriptions.items():
        path = CODES / "figures" / "revision_v2" / "point5" / (name + ".png")
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            rows.append({"name": name, "path": _relative(path), "status": "ready_point5",
                         "role": description, "width_px": image.width, "height_px": image.height,
                         "dpi_x": float(dpi[0]), "dpi_y": float(dpi[1]),
                         "source": "point5 raw diagnostics"})
    for number in (1, 2):
        path = CODES / "figures" / ("figure_{0}.png".format(number))
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            rows.append({"name": "legacy_figure_{0}".format(number), "path": _relative(path),
                         "status": "legacy_review_required", "role": "existing article figure",
                         "width_px": image.width, "height_px": image.height,
                         "dpi_x": float(dpi[0]), "dpi_y": float(dpi[1]),
                         "source": "pre-point6 material"})
    return rows


def _table_specs(point2: dict[str, Any], provenance: dict[str, Any],
                 point4: dict[str, Any], point5: dict[str, Any]) -> dict[str, dict[str, Any]]:
    post_fields = ["dataset", "scenario", "uses_ground_truth", "exclusion_reason",
                   "segments_retained", "segments_total", "segment_fraction",
                   "clusters_retained", "clusters_total", "prediction_outside_events",
                   "prediction_outside_affected_clusters",
                   "prediction_points_outside_before_processing"] + [
                       metric + "_cluster_equal_mean" for metric in ABS_METRICS]
    diagnostic_fields = ["dataset", "pressure", "clusters"]
    for metric in ("pearson_r", "spearman_rho", "r2", "calibration_slope",
                   "calibration_intercept"):
        diagnostic_fields.extend([metric, metric + "_ci95_lower", metric + "_ci95_upper"])
    diagnostic_fields.extend(["ba_bias", "ba_bias_ci95_lower", "ba_bias_ci95_upper",
                              "ba_traditional_lower", "ba_traditional_upper",
                              "ba_nonparametric_lower", "ba_nonparametric_upper",
                              "ba_nonparametric_lower_ci95_lower",
                              "ba_nonparametric_lower_ci95_upper",
                              "ba_nonparametric_upper_ci95_lower",
                              "ba_nonparametric_upper_ci95_upper"])
    return {
        "dataset_composition": {"fields": ["dataset", "role", "input_entities",
            "loaded_entities", "represented_clusters", "evaluated_segments",
            "initial_excluded_segments", "cluster_excluded_segments", "notes"],
            "rows": _composition_rows(provenance, point4, point5)},
        "main_metrics": {"fields": ["dataset", "metric", "estimand", "unit", "n",
            "mean", "sd", "ci95_lower", "ci95_upper"], "rows": _main_metric_rows(point5)},
        "signed_errors": {"fields": ["dataset", "metric", "estimand", "unit", "n",
            "mean", "sd", "ci95_lower", "ci95_upper"], "rows": _signed_rows(point5)},
        "bhs_proportions": {"fields": ["dataset", "pressure", "threshold_mmhg", "estimand",
            "percent", "sd", "n", "ci95_lower", "ci95_upper"], "rows": _bhs_rows(point5)},
        "normalization_holdout": {"fields": ["row_type", "variant", "role", "comparator",
            "metric", "estimate", "ci95_lower", "ci95_upper", "p_holm", "cliffs_delta"],
            "rows": _normalization_rows(point4, point5)},
        "postprocessing": {"fields": post_fields, "rows": _postprocessing_rows(point5)},
        "robust_tails": {"fields": ["dataset", "scenario", "metric", "q1", "median", "q3",
            "p90", "p95", "p99", "p99_5", "maximum", "count_gt30", "count_gt50",
            "count_gt100", "count_gt200", "count_gt500"], "rows": _tail_rows(point5)},
        "correlation_agreement": {"fields": diagnostic_fields,
            "rows": _diagnostic_rows(point5)},
        "calibration_bins": {"fields": ["dataset", "pressure", "lower", "upper", "segments",
            "clusters", "shown", "reference_mean", "prediction_mean"],
            "rows": _calibration_rows(point5)},
        "legacy_audits": {"fields": ["audit_type", "metric", "subset", "n", "mean", "sd",
            "paper", "reproduced", "deviation"], "rows": _legacy_rows(point2, point5)},
        "figure_index": {"fields": ["name", "path", "status", "role", "width_px", "height_px",
            "dpi_x", "dpi_y", "source"], "rows": _figure_rows()},
    }


def _full_status(input_hashes: dict[str, str]) -> str:
    if not FULL_VERIFY_PATH.exists():
        return "not_run"
    log = read_json(FULL_VERIFY_PATH)
    if log.get("status") == "passed" and log.get("inputs") == input_hashes:
        return "passed"
    return "stale"


def build_bundle() -> dict[str, Any]:
    _validate_nested_integrity()
    point2, provenance, point4, point5 = _load_sources()
    _cross_validate(point2, provenance, point4, point5)
    hashes = _input_hashes()
    tables = _table_specs(point2, provenance, point4, point5)
    bundle = {
        "schema_version": 2,
        "analysis": "revision_point6_canonical_article_materials",
        "generation": "deterministic aggregation; no inference or acquisition",
        "units": "mmHg unless explicitly percent",
        "primary_scenario": "raw",
        "primary_estimand": "mean within cluster, then equal weight per cluster",
        "seed": 20260908,
        "bootstrap_replicates": 10000,
        "full_verification": _full_status(hashes),
        "source_precedence": [
            "point5: final metrics, post-processing, correlations, agreement, figures",
            "point4: holdout comparisons and normalization roles",
            "provenance: dataset composition",
            "point2: legacy x200 reproduction only",
        ],
        "sources": _source_map(),
        "tables": tables,
        "limitations": point5["limitations"] + [
            "Existing article figures 1/2 are legacy and require review during point 7.",
            "This bundle does not modify or supersede the manuscript by itself.",
        ],
    }
    failures = find_forbidden_keys(bundle)
    if failures:
        raise AssertionError("public bundle contains identifier keys: " + ", ".join(failures))
    return bundle


def _table_payloads(bundle: dict[str, Any]) -> dict[Path, bytes]:
    payloads = {}
    for name, table in bundle["tables"].items():
        payloads[TABLE_ROOT / (name + ".csv")] = csv_bytes(table["rows"], table["fields"])
        payloads[TABLE_ROOT / (name + ".md")] = markdown_bytes(table["rows"], table["fields"])
    return payloads


def _notebook(bundle: dict[str, Any]) -> dict[str, Any]:
    inventory = ["{0}: {1} rows".format(name, len(table["rows"]))
                 for name, table in bundle["tables"].items()]
    figures = [row["path"] for row in bundle["tables"]["figure_index"]["rows"]
               if row["status"] == "ready_point5"]
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Article revision v2 — canonical verified materials\n",
                "This notebook loads only the public point6 bundle, derived tables, and ready PNG files."
            ]},
            {"cell_type": "code", "execution_count": 1, "metadata": {},
             "source": ["import json\n", "from pathlib import Path\n",
                        "root = Path.cwd().parent if Path.cwd().name == 'codes' else Path.cwd()\n",
                        "bundle = json.loads((root / 'revision_v2/point6/article_bundle_v2.json').read_text(encoding='utf-8'))\n",
                        "print(bundle['analysis'], bundle['full_verification'])"],
             "outputs": [{"name": "stdout", "output_type": "stream", "text": [
                 bundle["analysis"] + " " + bundle["full_verification"] + "\n"]}]},
            {"cell_type": "code", "execution_count": 2, "metadata": {},
             "source": ["for name, table in bundle['tables'].items():\n",
                        "    print(f\"{name}: {len(table['rows'])} rows\")"],
             "outputs": [{"name": "stdout", "output_type": "stream",
                          "text": [line + "\n" for line in inventory]}]},
            {"cell_type": "code", "execution_count": 3, "metadata": {},
             "source": ["from IPython.display import Image\n",
                        "ready_figures = []\n",
                        "for row in bundle['tables']['figure_index']['rows']:\n",
                        "    if row['status'] == 'ready_point5':\n",
                        "        ready_figures.append(Image(filename=str(root / row['path'])))\n",
                        "        print(row['path'])"],
             "outputs": [{"name": "stdout", "output_type": "stream",
                          "text": [path + "\n" for path in figures]}]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10+"}},
        "nbformat": 4, "nbformat_minor": 4,
    }


def _report(bundle: dict[str, Any]) -> bytes:
    status = ("**выполнен и полностью верифицирован**" if bundle["full_verification"] == "passed"
              else "**собран; full verify ещё не выполнен для текущих входов**")
    lines = ["# Ревизия, пункт 6: воспроизводимый контур", "",
             "Статус: " + status + ".", "",
             "Исходные исследовательские ноутбуки и скрипты не изменялись. Point6 детерминированно собирает публичные материалы из агрегатов пунктов 2–5 и не содержит команд инференса или выгрузки данных.", "",
             "## Авторитетность источников", "",
             "1. Point5 — итоговые raw/postprocessing метрики, корреляции, agreement и диагностические рисунки.",
             "2. Point4 — независимый holdout, comparisons и роль вариантов нормализации.",
             "3. Provenance summary — состав выборок.",
             "4. Point2 — только аудит физической шкалы и воспроизведение ×200.", "",
             "## Интерфейс", "", "```text",
             "python -m revision_v2.point6 build",
             "python -m revision_v2.point6 verify",
             "python -m revision_v2.point6 verify --full", "```", "",
             "Quick verify проверяет схемы, внутренние манифесты целостности, межэтапное совпадение и детерминированную пересборку публичных файлов. Full verify дополнительно запускает существующие проверки point2, provenance, point4 и point5 после no-inference preflight.", "",
             "## Article-ready таблицы", "", "| Table | Rows | CSV | Markdown |", "|---|---:|---|---|"]
    for name, table in bundle["tables"].items():
        lines.append("| `{0}` | {1} | `{2}` | `{3}` |".format(
            name, len(table["rows"]), _relative(TABLE_ROOT / (name + ".csv")),
            _relative(TABLE_ROOT / (name + ".md"))))
    lines.extend(["", "Канонический машинный источник: `{0}`. CSV, Markdown и notebook являются только его производными представлениями.".format(_relative(BUNDLE_PATH)), "",
                  "## Рисунки", "", "| Name | Status | Size | DPI |", "|---|---|---:|---:|"])
    for row in bundle["tables"]["figure_index"]["rows"]:
        lines.append("| `{0}` | {1} | {2}×{3} | {4:.1f} |".format(
            row["name"], row["status"], row["width_px"], row["height_px"],
            row["dpi_x"]))
    lines.extend(["", "Legacy figures 1/2 не копировались и не объявляются готовыми: их соответствие новым таблицам проверяется в пункте 7.", "",
                  "## Источники", "", "| Source | Role |", "|---|---|"])
    for source in bundle["sources"]:
        lines.append("| `{0}` | {1} |".format(source["path"], source["role"]))
    lines.extend(["", "## Контроль воспроизводимости", "",
                  "Исходные агрегаты, веса и кэши предсказаний проверяются по локальным манифестам в игнорируемом `codes/data/`. Публичные файлы проверяются детерминированной пересборкой, структурой таблиц, числовыми инвариантами, размерами рисунков и отсутствием идентификаторов.", "",
                  "## Граница этапа", "",
                  "Point6 организует воспроизводимость и передачу чисел в point7. Markdown-рукопись и DOCX на этом этапе не изменяются; публикация и редакционный перенос не выполняются.", ""])
    return ("\n".join(lines)).encode("utf-8")


def _write_artifacts() -> dict[str, Any]:
    bundle = build_bundle()
    bundle_payload = pretty_json_bytes(bundle)
    table_payloads = _table_payloads(bundle)
    notebook_payload = pretty_json_bytes(_notebook(bundle))
    write_if_changed(BUNDLE_PATH, bundle_payload)
    for path, payload in table_payloads.items():
        write_if_changed(path, payload)
    write_if_changed(NOTEBOOK_PATH, notebook_payload)
    report_payload = _report(bundle)
    write_if_changed(REPORT_PATH, report_payload)
    integrity = {"schema_version": 2, "inputs": _input_hashes(),
                 "prediction_caches": _prediction_hashes(),
                 "note": "local-only input and prediction-cache integrity; public outputs are regenerated"}
    write_if_changed(INTEGRITY_PATH, pretty_json_bytes(integrity))
    return bundle


def build_command(args: argparse.Namespace) -> None:
    bundle = _write_artifacts()
    print("Built canonical bundle ({0}), {1} tables, no inference".format(
        bundle["full_verification"], len(bundle["tables"])))


def _verify_public_artifacts() -> dict[str, Any]:
    bundle = build_bundle()
    expected_bundle = pretty_json_bytes(bundle)
    if BUNDLE_PATH.read_bytes() != expected_bundle:
        raise AssertionError("canonical bundle differs from regenerated content")
    table_payloads = _table_payloads(bundle)
    for path, expected in table_payloads.items():
        if path.read_bytes() != expected:
            raise AssertionError("derived table differs: " + _relative(path))
    expected_notebook = pretty_json_bytes(_notebook(bundle))
    if NOTEBOOK_PATH.read_bytes() != expected_notebook:
        raise AssertionError("thin notebook differs")
    if REPORT_PATH.read_bytes() != _report(bundle):
        raise AssertionError("point6 report differs")
    expected_integrity = {"schema_version": 2, "inputs": _input_hashes(),
                          "prediction_caches": _prediction_hashes(),
                          "note": "local-only input and prediction-cache integrity; public outputs are regenerated"}
    if read_json(INTEGRITY_PATH) != expected_integrity:
        raise AssertionError("point6 integrity differs")
    public_text = (BUNDLE_PATH.read_text(encoding="utf-8") +
                   NOTEBOOK_PATH.read_text(encoding="utf-8") +
                   REPORT_PATH.read_text(encoding="utf-8") +
                   "".join(path.read_text(encoding="utf-8")
                           for path in table_payloads))
    for token in ("case_id", "subject_id", "record_id", "record_idx", "cluster_id"):
        if token in public_text:
            raise AssertionError("public artifact exposes identifier field: " + token)
    for row in bundle["tables"]["figure_index"]["rows"]:
        if row["status"] == "ready_point5" and min(row["dpi_x"], row["dpi_y"]) < 299:
            raise AssertionError("ready figure is below 300 dpi: " + row["name"])
    return bundle


def _prediction_hashes() -> dict[str, str]:
    point4 = read_json(POINT4_INTEGRITY)["artifacts"]
    keys = sorted(key for key in point4 if key.startswith("prediction_") or
                  key.startswith("old_prediction_"))
    return {key: point4[key] for key in keys}


def _supports_point2(python_path: Path) -> bool:
    completed = subprocess.run(
        [str(python_path), "-c", "import sys,h5py; assert sys.version_info >= (3,10)"],
        cwd=str(ROOT), text=True, capture_output=True, check=False)
    return completed.returncode == 0


def _analysis_python(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    environment = os.environ.get("PPG2ABP_ANALYSIS_PYTHON")
    if environment:
        candidates.append(Path(environment))
    candidates.append(Path(sys.executable))
    executable = Path(sys.executable).resolve()
    if executable.parent.parent.name == "envs":
        candidates.append(executable.parent.parent / "marl" / "python.exe")
    else:
        candidates.append(executable.parent / "envs" / "marl" / "python.exe")
    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        if candidate.is_file() and _supports_point2(candidate):
            return candidate
    raise RuntimeError(
        "point2 full verification needs Python >=3.10 with h5py; pass "
        "--analysis-python or set PPG2ABP_ANALYSIS_PYTHON")


def _run_full_verifiers(inputs: dict[str, str], analysis_python: Path) -> dict[str, Any]:
    commands = [
        ("point2", [str(analysis_python), "-m",
                    "revision_v2.point6.point2_verify_adapter"]),
        ("provenance", [sys.executable, str(CODES / "segment_provenance.py"), "verify"]),
        ("point4", [sys.executable, "-m", "revision_v2.point4", "verify"]),
        ("point5", [sys.executable, "-m", "revision_v2.point5", "verify"]),
    ]
    prediction_before = _prediction_hashes()
    results = []
    for name, command in commands:
        print("FULL VERIFY: " + name, flush=True)
        completed = subprocess.run(command, cwd=str(ROOT), text=True,
                                   capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError("{0} verify failed:\n{1}\n{2}".format(
                name, completed.stdout, completed.stderr))
        if name == "point2":
            payload = json.loads(completed.stdout)
            if set(payload["cache_status"].values()) != {"cache"}:
                raise AssertionError("point2 verification did not use only caches")
        results.append({"name": name, "returncode": completed.returncode})
        print("FULL VERIFY OK: " + name, flush=True)
    _validate_nested_integrity()
    if _prediction_hashes() != prediction_before:
        raise AssertionError("prediction integrity map changed during full verify")
    if _input_hashes() != inputs:
        raise AssertionError("aggregate inputs changed during full verify")
    return {"schema_version": 1, "status": "passed", "inputs": inputs,
            "prediction_hashes": prediction_before, "commands": results,
            "inference": "not run; point2 statuses were cache and all cache hashes stayed fixed"}


def verify_command(args: argparse.Namespace) -> None:
    _validate_nested_integrity()
    if not args.full:
        bundle = _verify_public_artifacts()
        print("QUICK VERIFY OK: schemas, internal integrity, cross-stage invariants, bundle, tables, "
              "figures, report, and notebook ({0})".format(bundle["full_verification"]))
        return
    # Abort before invoking point2 if point4/5 integrity or current public inputs differ.
    _verify_public_artifacts()
    inputs = _input_hashes()
    analysis_python = _analysis_python(args.analysis_python)
    print("Point2 analysis Python: " + str(analysis_python), flush=True)
    log = _run_full_verifiers(inputs, analysis_python)
    write_if_changed(FULL_VERIFY_PATH, pretty_json_bytes(log))
    _write_artifacts()
    bundle = _verify_public_artifacts()
    if bundle["full_verification"] != "passed":
        raise AssertionError("full verification status was not promoted")
    print("FULL VERIFY OK: all existing verification layers passed; prediction caches unchanged")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.set_defaults(function=build_command)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--full", action="store_true",
                        help="run point2/provenance/point4/point5 verification after preflight")
    verify.add_argument("--analysis-python",
                        help="Python >=3.10 with h5py for the legacy point2 verifier")
    verify.set_defaults(function=verify_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)
