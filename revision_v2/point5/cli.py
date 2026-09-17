"""Authoritative CLI for revision point 5.

Run from the repository root::

    python -m revision_v2.point5 analyze
    python -m revision_v2.point5 figures
    python -m revision_v2.point5 report
    python -m revision_v2.point5 verify

No command imports a model framework or performs inference.  All calculations
start from the immutable prediction caches documented by revision point 4.
"""

from __future__ import division, print_function

import argparse
import csv
import json
import os

import numpy as np

from revision_v2.point4.core import (ABP_MIN, ABP_SCALE, assert_nested_equal,
                                     load_pickle_compat, read_json, sha256_file,
                                     write_json)

from . import __version__
from .core import (ABS_METRICS, BOOTSTRAP_REPLICATES, METRICS, SCENARIOS, SEED,
                   bland_altman_statistics, calibration_bins, cluster_means,
                   correlation_statistics, pair_statistics, scenario_masks,
                   scenario_statistics)


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
CODES = os.path.join(ROOT, "codes")
DATA = os.path.join(CODES, "data")
POINT4 = os.path.join(DATA, "revision_v2", "point4", "v1")
PRIVATE_ROOT = os.path.join(DATA, "revision_v2", "point5", "v1")
FEATURE_DIR = os.path.join(PRIVATE_ROOT, "features")
RESULTS_PATH = os.path.join(PRIVATE_ROOT, "results.json")
SOURCE_MANIFEST = os.path.join(PRIVATE_ROOT, "source_integrity.json")
INTEGRITY_PATH = os.path.join(PRIVATE_ROOT, "integrity.json")
FIGURE_DIR = os.path.join(CODES, "figures", "revision_v2", "point5")
REPORT_PATH = os.path.join(ROOT, ".context", "REVISION_POINT5.md")
NOTEBOOK_PATH = os.path.join(CODES, "_revision_point5.ipynb")
POINT4_RESULTS = os.path.join(POINT4, "results.json")
POINT4_INTEGRITY = os.path.join(POINT4, "integrity.json")
PROVENANCE = os.path.join(DATA, "provenance")
OLD_CACHE = os.path.join(DATA, "unified_evaluation_cache")
HOLDOUT_SIGNALS = os.path.join(POINT4, "acquisition", "holdout_signals.npz")
HOLDOUT_SEGMENTS = os.path.join(POINT4, "acquisition", "segment_manifest.csv")

DATASETS = ("mimic", "abp_ppg", "vitaldb_old", "holdout_A",
            "holdout_B", "holdout_C")
MAIN_DATASETS = ("mimic", "abp_ppg", "vitaldb_old", "holdout_A")
LABELS = {"mimic": "MIMIC corrected", "abp_ppg": "ABP_PPG",
          "vitaldb_old": "VitalDB original A", "holdout_A": "VitalDB holdout A",
          "holdout_B": "Holdout B", "holdout_C": "Holdout C"}
PREDICTION_REFERENCE_ZOOM = {"sbp": (60.0, 220.0),
                             "dbp": (20.0, 140.0)}

FEATURE_FIELDS = (
    "segment_index", "cluster_id", "cluster_included", "provenance_status",
    "gt_min", "gt_max", "gt_mean", "gt_range", "raw_pred_min", "raw_pred_max",
    "raw_pred_mean", "raw_pred_range", "clip_pred_min", "clip_pred_max",
    "clip_pred_mean", "clip_pred_range", "pred_outside_20_300",
    "pred_points_outside_20_300", "gt_sbp", "gt_dbp", "gt_map",
    "raw_pred_sbp", "raw_pred_dbp", "raw_pred_map", "clip_pred_sbp",
    "clip_pred_dbp", "clip_pred_map", "raw_waveform", "raw_sbp_abs",
    "raw_dbp_abs", "raw_map_abs", "raw_sbp_signed", "raw_dbp_signed",
    "raw_map_signed", "clip_waveform", "clip_sbp_abs", "clip_dbp_abs",
    "clip_map_abs", "clip_sbp_signed", "clip_dbp_signed", "clip_map_signed")


def _mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _relative(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def _outside_plot_limits(reference, prediction, limits):
    lower, upper = limits
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return ((reference < lower) | (reference > upper) |
            (prediction < lower) | (prediction > upper))


def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _feature_path(dataset):
    return os.path.join(FEATURE_DIR, dataset + ".csv")


def _prediction_paths(dataset):
    if dataset.startswith("holdout_"):
        variant = dataset[-1]
        return (os.path.join(POINT4, "predictions", "holdout_{0}_pred_norm.npy".format(variant)),
                os.path.join(POINT4, "predictions", "holdout_{0}_manifest.json".format(variant)))
    cache_name = "vitaldb" if dataset == "vitaldb_old" else dataset
    return (os.path.join(OLD_CACHE, cache_name + "_pred_norm.npy"),
            os.path.join(OLD_CACHE, cache_name + "_manifest.json"))


def _input_paths():
    paths = {
        "point4_integrity": POINT4_INTEGRITY, "point4_results": POINT4_RESULTS,
        "source_mimic": os.path.join(DATA, "test_meta9_corrected.p"),
        "source_abp_ppg": os.path.join(DATA, "abp_ppg_test.p"),
        "source_vitaldb_old": os.path.join(CODES, "raw_data", "vitaldb_raw.p"),
        "source_holdout": HOLDOUT_SIGNALS,
        "provenance_mimic": os.path.join(PROVENANCE, "mimic_segments.csv"),
        "provenance_abp_ppg": os.path.join(PROVENANCE, "abp_ppg_segments.csv"),
        "provenance_vitaldb_old": os.path.join(PROVENANCE, "vitaldb_segments.csv"),
        "provenance_holdout": HOLDOUT_SEGMENTS,
        "provenance_summary": os.path.join(PROVENANCE, "summary.json"),
        "point4_acquisition_manifest": os.path.join(POINT4, "acquisition", "manifest.json"),
    }
    for dataset in DATASETS:
        pred, manifest = _prediction_paths(dataset)
        paths["prediction_" + dataset] = pred
        paths["prediction_manifest_" + dataset] = manifest
        paths["point4_errors_" + dataset] = os.path.join(POINT4, "errors", dataset + ".csv")
    return paths


def _current_input_hashes():
    return dict((name, sha256_file(path)) for name, path in _input_paths().items())


def _validate_point4():
    # Reuse point 4's own complete artifact map, including its sources, weights,
    # manifests, predictions, error tables, and executable provenance.
    from revision_v2.point4.cli import _artifact_hashes
    frozen = read_json(POINT4_INTEGRITY)["artifacts"]
    actual = _artifact_hashes(True)
    if frozen != actual:
        changed = sorted(set(key for key in set(frozen) | set(actual)
                             if frozen.get(key) != actual.get(key)))
        raise AssertionError("point 4 integrity failure: {0}".format(", ".join(changed)))
    for dataset in DATASETS:
        pred_path, manifest_path = _prediction_paths(dataset)
        manifest = read_json(manifest_path)
        prediction = np.load(pred_path, mmap_mode="r")
        expected_shape = manifest.get("prediction_shape", manifest.get("input_shape"))
        if list(prediction.shape) != list(expected_shape[:2]):
            raise AssertionError("prediction shape mismatch: " + dataset)
        if manifest.get("prediction_sha256") not in (None, sha256_file(pred_path)):
            raise AssertionError("manifest prediction hash mismatch: " + dataset)
        if dataset.startswith("holdout_"):
            if manifest["holdout_signals_sha256"] != sha256_file(HOLDOUT_SIGNALS):
                raise AssertionError("holdout source hash mismatch: " + dataset)
        else:
            source = os.path.join(ROOT, manifest["source"].replace("\\", os.sep).replace("/", os.sep))
            if manifest["source_sha256"] != sha256_file(source):
                raise AssertionError("source hash mismatch: " + dataset)
    return frozen


def _cluster_rows(dataset, count):
    if dataset == "mimic":
        rows = [row for row in _read_csv(os.path.join(PROVENANCE, "mimic_segments.csv"))
                if row["split_role"] == "test"]
        result = []
        for index, row in enumerate(rows):
            included = row["status"] in ("exact_unique", "exact_record_only")
            result.append((index, ("{0}:{1}".format(row["part"], row["record_idx"])
                                   if included else ""), int(included), row["status"]))
    elif dataset == "abp_ppg":
        rows = [row for row in _read_csv(os.path.join(PROVENANCE, "abp_ppg_segments.csv"))
                if row["accepted"] == "1"]
        rows.sort(key=lambda row: int(row["evaluation_index"]))
        result = [(index, row["subject_id"], 1, "accepted")
                  for index, row in enumerate(rows)]
    elif dataset == "vitaldb_old":
        rows = _read_csv(os.path.join(PROVENANCE, "vitaldb_segments.csv"))
        rows.sort(key=lambda row: int(row["segment_id"]))
        result = [(index, row["subject_id"], 1, row["status"])
                  for index, row in enumerate(rows)]
    else:
        rows = _read_csv(HOLDOUT_SEGMENTS)
        rows.sort(key=lambda row: int(row["segment_index"]))
        result = [(index, row["subject_id"], 1, "accepted")
                  for index, row in enumerate(rows)]
    if len(result) != count:
        raise AssertionError("provenance length mismatch: " + dataset)
    return result


def _ground_truth(dataset):
    if dataset == "mimic":
        data = load_pickle_compat(os.path.join(DATA, "test_meta9_corrected.p"))
        return np.asarray(data["Y_test"][:, :, 0], dtype=np.float64) * ABP_SCALE + ABP_MIN
    if dataset == "abp_ppg":
        data = load_pickle_compat(os.path.join(DATA, "abp_ppg_test.p"))
        return np.asarray(data["Y_test"][:, :, 0], dtype=np.float64)
    if dataset == "vitaldb_old":
        data = load_pickle_compat(os.path.join(CODES, "raw_data", "vitaldb_raw.p"))
        return np.asarray(data["Y_raw"], dtype=np.float64)
    holdout = np.load(HOLDOUT_SIGNALS, mmap_mode="r")
    return np.asarray(holdout["Y_raw"], dtype=np.float64)


def _error_vectors(prediction, ground_truth):
    signed_sbp = np.max(prediction, axis=1) - np.max(ground_truth, axis=1)
    signed_dbp = np.min(prediction, axis=1) - np.min(ground_truth, axis=1)
    signed_map = np.mean(prediction, axis=1) - np.mean(ground_truth, axis=1)
    return {"waveform": np.mean(np.abs(prediction - ground_truth), axis=1),
            "sbp_abs": np.abs(signed_sbp), "dbp_abs": np.abs(signed_dbp),
            "map_abs": np.abs(signed_map), "sbp_signed": signed_sbp,
            "dbp_signed": signed_dbp, "map_signed": signed_map}


def _write_feature_table(dataset):
    pred_path, _ = _prediction_paths(dataset)
    pred_norm = np.load(pred_path, mmap_mode="r")
    gt = _ground_truth(dataset)
    if pred_norm.shape != gt.shape:
        raise AssertionError("prediction/GT mismatch: " + dataset)
    provenance = _cluster_rows(dataset, len(gt))
    path = _feature_path(dataset)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for start in range(0, len(gt), 512):
            stop = min(start + 512, len(gt))
            truth = gt[start:stop]
            raw = np.asarray(pred_norm[start:stop], dtype=np.float64) * ABP_SCALE + ABP_MIN
            clipped = np.clip(raw, 20.0, 300.0)
            raw_errors = _error_vectors(raw, truth)
            clip_errors = _error_vectors(clipped, truth)
            features = {
                "gt_min": np.min(truth, axis=1), "gt_max": np.max(truth, axis=1),
                "gt_mean": np.mean(truth, axis=1), "gt_range": np.ptp(truth, axis=1),
                "raw_pred_min": np.min(raw, axis=1), "raw_pred_max": np.max(raw, axis=1),
                "raw_pred_mean": np.mean(raw, axis=1), "raw_pred_range": np.ptp(raw, axis=1),
                "clip_pred_min": np.min(clipped, axis=1),
                "clip_pred_max": np.max(clipped, axis=1),
                "clip_pred_mean": np.mean(clipped, axis=1),
                "clip_pred_range": np.ptp(clipped, axis=1),
                "pred_outside_20_300": np.any((raw < 20.0) | (raw > 300.0), axis=1),
                "pred_points_outside_20_300": np.sum((raw < 20.0) | (raw > 300.0), axis=1),
                "gt_sbp": np.max(truth, axis=1), "gt_dbp": np.min(truth, axis=1),
                "gt_map": np.mean(truth, axis=1), "raw_pred_sbp": np.max(raw, axis=1),
                "raw_pred_dbp": np.min(raw, axis=1), "raw_pred_map": np.mean(raw, axis=1),
                "clip_pred_sbp": np.max(clipped, axis=1),
                "clip_pred_dbp": np.min(clipped, axis=1),
                "clip_pred_map": np.mean(clipped, axis=1),
            }
            for name in METRICS:
                features["raw_" + name] = raw_errors[name]
                features["clip_" + name] = clip_errors[name]
            for offset in range(stop - start):
                index, cluster, included, status = provenance[start + offset]
                row = {"segment_index": index, "cluster_id": cluster,
                       "cluster_included": included, "provenance_status": status}
                for key, vector in features.items():
                    value = vector[offset]
                    if isinstance(value, (bool, np.bool_)):
                        row[key] = int(value)
                    elif isinstance(value, (int, np.integer)):
                        row[key] = int(value)
                    else:
                        row[key] = format(float(value), ".17g")
                writer.writerow(row)
    os.replace(temporary, path)
    return path


def _load_features(dataset):
    rows = _read_csv(_feature_path(dataset))
    string_fields = ("cluster_id", "provenance_status")
    int_fields = ("segment_index", "cluster_included", "pred_outside_20_300",
                  "pred_points_outside_20_300")
    columns = {}
    for field in FEATURE_FIELDS:
        if field in string_fields:
            columns[field] = np.asarray([row[field] for row in rows])
        elif field in int_fields:
            columns[field] = np.asarray([int(row[field]) for row in rows], dtype=np.int64)
        else:
            columns[field] = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return columns


def _raw_match(results):
    point4 = read_json(POINT4_RESULTS)["datasets"]
    checks = {}
    for dataset in DATASETS:
        current = results["datasets"][dataset]["scenarios"]["raw"]
        previous = point4[dataset]
        for metric in METRICS:
            expected_segment = (previous["full_segment_weighted"][metric]
                                if dataset == "mimic"
                                else previous["metrics"][metric]["segment_weighted"])
            actual_segment = current["metrics"][metric]["segment_weighted"]
            for key in ("mean", "sd"):
                if not np.isclose(actual_segment[key], expected_segment[key], rtol=0, atol=1e-12):
                    raise AssertionError("raw point4 mismatch: {0}/{1}/{2}".format(
                        dataset, metric, key))
            expected_cluster = previous["metrics"][metric]["cluster_equal"]
            actual_cluster = current["metrics"][metric]["cluster_equal"]
            for key in ("mean", "sd"):
                if not np.isclose(actual_cluster[key], expected_cluster[key], rtol=0, atol=1e-12):
                    raise AssertionError("cluster raw point4 mismatch: {0}/{1}/{2}".format(
                        dataset, metric, key))
        checks[dataset] = True
    return checks


def _diagnostics(columns):
    included = np.asarray(columns["cluster_included"], dtype=bool)
    clusters = np.asarray(columns["cluster_id"])[included]
    result = {"correlations": {}, "calibration": {}, "bland_altman": {}}
    for pressure in ("sbp", "dbp"):
        reference = np.asarray(columns["gt_" + pressure], dtype=np.float64)[included]
        prediction = np.asarray(columns["raw_pred_" + pressure], dtype=np.float64)[included]
        _, cluster_reference = cluster_means(reference, clusters)
        _, cluster_prediction = cluster_means(prediction, clusters)
        result["correlations"][pressure] = {
            "cluster_level_primary": correlation_statistics(
                cluster_reference, cluster_prediction, seed=SEED,
                replicates=BOOTSTRAP_REPLICATES),
            "segment_level_descriptive": pair_statistics(reference, prediction),
        }
        result["calibration"][pressure] = calibration_bins(
            reference, prediction, clusters, minimum_clusters=10)
        result["bland_altman"][pressure] = bland_altman_statistics(
            prediction - reference, clusters, seed=SEED,
            replicates=BOOTSTRAP_REPLICATES)
    return result


def calculate_results():
    datasets = {}
    feature_cache = {}
    for dataset in DATASETS:
        columns = _load_features(dataset)
        feature_cache[dataset] = columns
        masks, p95 = scenario_masks(columns)
        scenarios = {}
        for scenario in SCENARIOS:
            stats = scenario_statistics(columns, scenario, masks[scenario], seed=SEED,
                                        replicates=BOOTSTRAP_REPLICATES)
            excluded = int(len(masks[scenario]) - np.sum(masks[scenario]))
            reason = {"raw": "none", "clip": "none (values clipped, no rejection)",
                      "reject": "prediction outside [20,300]",
                      "oracle_top5": "raw waveform MAE above dataset P95",
                      "gt_50_200": "GT min <50 or GT max >200"}[scenario]
            stats["exclusion"] = {"reason": reason, "segments": excluded,
                                  "uses_ground_truth": scenario in ("oracle_top5", "gt_50_200")}
            stats["prediction_points_outside_20_300_before_processing"] = int(
                np.sum(columns["pred_points_outside_20_300"][masks[scenario]]))
            scenarios[scenario] = stats
        datasets[dataset] = {"label": LABELS[dataset], "waveform_mae_p95": p95,
                             "scenarios": scenarios,
                             "diagnostics": (_diagnostics(columns)
                                             if dataset in MAIN_DATASETS else None),
                             "role": ("main" if dataset in MAIN_DATASETS
                                      else "secondary normalization ablation only")}
    result = {
        "schema_version": 1,
        "analysis": "revision_point5_outliers_postprocessing_diagnostics",
        "seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_scenario": "raw", "datasets": datasets,
        "scenario_definitions": {
            "raw": "all originally eligible segments; sole primary result",
            "clip": "pointwise prediction clipping to [20,300] mmHg; retention 100%",
            "reject": "reject if any prediction sample is outside [20,300] mmHg; GT-blind",
            "oracle_top5": "exclude raw waveform MAE above dataset-specific P95; truth-dependent diagnostic",
            "gt_50_200": "GT min >=50 and GT max <=200; truth-dependent subgroup",
        },
        "estimands": {"segment_weighted": "each segment has equal weight",
                       "cluster_equal": "mean within cluster, then equal cluster weight"},
        "old_text_audit": {
            "dataset": "vitaldb_old", "metric": "raw waveform MAE",
            "claimed_mixed_value": "22.25 +/- 5.78",
            "explanation": "mean and SD came from different subsets",
        },
        "limitations": [
            "Correlation measures ranking/association, not accuracy or agreement.",
            "Limits of agreement are population limits, not confidence intervals for one patient.",
            "BHS/AAMI-style summaries do not constitute device certification.",
            "Oracle top-5% exclusion uses truth and cannot be deployed as post-processing.",
            "Holdout B/C are secondary ablations and were not used to reselect normalization.",
        ],
    }
    vital = feature_cache["vitaldb_old"]
    masks, _ = scenario_masks(vital)
    waveform = vital["raw_waveform"]
    intersection = masks["oracle_top5"] & masks["gt_50_200"]
    for name, mask in (("top5_retained", masks["oracle_top5"]),
                       ("gt_50_200", masks["gt_50_200"]),
                       ("intersection_audit_only", intersection)):
        result["old_text_audit"][name] = {
            "n": int(np.sum(mask)), "mean": float(np.mean(waveform[mask])),
            "sd": float(np.std(waveform[mask]))}
    result["raw_point4_exact_match"] = _raw_match(result)
    return result


def analyze_command(args):
    _mkdir(FEATURE_DIR)
    point4_artifacts = _validate_point4()
    inputs = _current_input_hashes()
    for dataset in DATASETS:
        path = _write_feature_table(dataset)
        print("Feature table: {0}".format(_relative(path)))
    result = calculate_results()
    write_json(RESULTS_PATH, result)
    manifest = {"schema_version": 1, "point4_artifacts": point4_artifacts,
                "inputs": inputs,
                "features": dict((dataset, sha256_file(_feature_path(dataset)))
                                 for dataset in DATASETS)}
    write_json(SOURCE_MANIFEST, manifest)
    print("Analysis saved; no model inference was run: {0}".format(_relative(RESULTS_PATH)))


def _figure_paths():
    names = ("error_distributions", "prediction_reference_full",
             "prediction_reference_zoom", "calibration",
             "bland_altman_full", "bland_altman_zoom")
    return dict((name, os.path.join(FIGURE_DIR, name + ".png")) for name in names)


def _save_figure(fig, path):
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")


def figures_command(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _mkdir(FIGURE_DIR)
    results = read_json(RESULTS_PATH)
    features = dict((name, _load_features(name)) for name in MAIN_DATASETS)
    colors = {"mimic": "#0072B2", "abp_ppg": "#009E73",
              "vitaldb_old": "#D55E00", "holdout_A": "#CC79A7"}
    paths = _figure_paths()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, dataset in zip(axes.flat, MAIN_DATASETS):
        values = np.sort(features[dataset]["raw_waveform"])
        axis.plot(values, np.arange(1, len(values) + 1) / len(values),
                  color=colors[dataset], linewidth=1.4)
        axis.set_xscale("symlog", linthresh=10)
        axis.set_title(LABELS[dataset])
        axis.set_xlabel("Waveform MAE (mmHg)")
        axis.set_ylabel("Empirical CDF")
        axis.grid(alpha=.2)
    fig.suptitle("Raw waveform-error distributions (all eligible segments)")
    fig.tight_layout(rect=(0, 0, 1, .96))
    _save_figure(fig, paths["error_distributions"]); plt.close(fig)

    for zoom in (False, True):
        fig, axes = plt.subplots(4, 2, figsize=(10, 15))
        for row, dataset in enumerate(MAIN_DATASETS):
            data = features[dataset]
            for column, pressure in enumerate(("sbp", "dbp")):
                axis = axes[row, column]
                x, y = data["gt_" + pressure], data["raw_pred_" + pressure]
                limits = PREDICTION_REFERENCE_ZOOM[pressure]
                outside_mask = _outside_plot_limits(x, y, limits)
                outside = int(np.sum(outside_mask))
                if zoom:
                    lower, upper = limits
                    axis.plot([lower, upper], [lower, upper], "--", color="black",
                              linewidth=.8)
                    visible = ~outside_mask
                    axis.hexbin(x[visible], y[visible], gridsize=55, bins="log", mincnt=1,
                                cmap="viridis", rasterized=True)
                    axis.set_xlim(lower, upper); axis.set_ylim(lower, upper)
                else:
                    identity = np.linspace(20.0, 300.0, 281)
                    axis.plot(identity, identity, "--", color="black", linewidth=.8)
                    axis.scatter(x, y, s=1.0, alpha=.18, color="#440154",
                                 edgecolors="none", rasterized=True)
                    axis.set_xlim(20, 300)
                    axis.set_yscale("symlog", linthresh=20)
                axis.set_title("{0} — {1}; outside zoom: {2}".format(
                    LABELS[dataset], pressure.upper(), outside))
                axis.set_xlabel("Reference (mmHg)"); axis.set_ylabel("Prediction (mmHg)")
        fig.suptitle("Raw prediction versus reference — {0}".format(
            "SBP 60–220; DBP 20–140 mmHg zoom" if zoom
            else "full range (prediction symlog)"))
        fig.tight_layout(rect=(0, 0, 1, .975))
        _save_figure(fig, paths["prediction_reference_zoom" if zoom else
                                "prediction_reference_full"]); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, dataset in zip(axes.flat, MAIN_DATASETS):
        for pressure, marker in (("sbp", "o"), ("dbp", "s")):
            bins = results["datasets"][dataset]["diagnostics"]["calibration"][pressure]
            shown = [row for row in bins if row["shown"]]
            axis.plot([row["reference_mean"] for row in shown],
                      [row["prediction_mean"] for row in shown], marker=marker,
                      linewidth=1.2, label=pressure.upper())
        axis.plot([20, 300], [20, 300], "--", color="black", linewidth=.8,
                  label="Identity")
        axis.set_xlim(20, 300); axis.set_ylim(20, 300)
        axis.set_title(LABELS[dataset]); axis.set_xlabel("Cluster-equal reference (mmHg)")
        axis.set_ylabel("Cluster-equal prediction (mmHg)"); axis.grid(alpha=.2)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    axes[0, 1].legend(handles, legend_labels, loc="upper right")
    fig.suptitle("Calibration in fixed 20-mmHg reference bins (≥10 clusters)")
    fig.tight_layout(rect=(0, 0, 1, .96))
    _save_figure(fig, paths["calibration"]); plt.close(fig)

    for zoom in (False, True):
        fig, axes = plt.subplots(4, 2, figsize=(10, 15))
        for row, dataset in enumerate(MAIN_DATASETS):
            data = features[dataset]
            for column, pressure in enumerate(("sbp", "dbp")):
                axis = axes[row, column]
                reference = data["gt_" + pressure]
                prediction = data["raw_pred_" + pressure]
                mean = (reference + prediction) / 2.0
                difference = prediction - reference
                ba = results["datasets"][dataset]["diagnostics"]["bland_altman"][pressure]
                axis.axhline(ba["bias"], color="#0072B2", linewidth=1.0, label="Bias")
                for limit in ba["nonparametric_limits"]:
                    axis.axhline(limit, color="#D55E00", linestyle="--", linewidth=1.0,
                                 label="P2.5/P97.5" if limit == ba["nonparametric_limits"][0] else None)
                outside = int(np.sum((mean < 20) | (mean > 300) |
                                     (difference < -100) | (difference > 100)))
                if zoom:
                    visible = ((mean >= 20) & (mean <= 300) &
                               (difference >= -100) & (difference <= 100))
                    axis.hexbin(mean[visible], difference[visible], gridsize=55,
                                bins="log", mincnt=1, cmap="magma", rasterized=True)
                    axis.set_xlim(20, 300); axis.set_ylim(-100, 100)
                else:
                    axis.scatter(mean, difference, s=1.0, alpha=.18, color="#440154",
                                 edgecolors="none", rasterized=True)
                    axis.set_xscale("symlog", linthresh=20)
                    axis.set_yscale("symlog", linthresh=20)
                axis.set_title("{0} — {1}; outside zoom: {2}".format(
                    LABELS[dataset], pressure.upper(), outside))
                axis.set_xlabel("Mean of prediction and reference (mmHg)")
                axis.set_ylabel("Prediction − reference (mmHg)")
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        axes[0, 1].legend(handles, legend_labels, loc="upper right")
        fig.suptitle("Raw Bland–Altman — {0}".format(
            "zoom" if zoom else "full range (symlog)"))
        fig.tight_layout(rect=(0, 0, 1, .975))
        _save_figure(fig, paths["bland_altman_zoom" if zoom else
                                "bland_altman_full"]); plt.close(fig)
    print("Saved six diagnostic figures at 320 dpi")


def _fmt(value):
    return format(float(value), ".3f")


def _scenario_table(results, scenario):
    lines = ["| Dataset | Retained segments / clusters | Waveform MAE | SBP abs. | DBP abs. | MAP abs. |",
             "|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        stats = results["datasets"][dataset]["scenarios"][scenario]
        coverage = stats["coverage"]
        values = []
        for metric in ABS_METRICS:
            item = stats["metrics"][metric]["cluster_equal"]
            values.append("{0} [{1}; {2}]".format(
                _fmt(item["mean"]), _fmt(item["ci95"][0]), _fmt(item["ci95"][1])))
        lines.append("| {0} | {1}/{2} / {3}/{4} | {5} | {6} | {7} | {8} |".format(
            LABELS[dataset], coverage["segments_retained"], coverage["segments_total"],
            coverage["clusters_retained"], coverage["clusters_total"], *values))
    return lines


def _render_report(results):
    audit = results["old_text_audit"]
    lines = ["# Ревизия, пункт 5: выбросы, постобработка и диагностика", "",
             "Статус расчёта: **выполнен и верифицирован**.", "",
             "Основными являются только результаты `raw`: все предсказания после исходного отбора пригодных входных сегментов сохранены, включая редкие взрывные значения. Новый инференс не запускался; анализ использует проверенные локальные кэши предсказаний пункта 4.", "",
             "## Аудит старого числа", "",
             "Старое `22,25 ± 5,78 мм рт. ст.` ошибочно объединяло среднее и SD из разных выборок. Воспроизводимые значения waveform MAE на исходном VitalDB A:", "",
             "- top-5% (оставлено N={0}): `{1} ± {2}`;".format(
                 audit["top5_retained"]["n"], _fmt(audit["top5_retained"]["mean"]),
                 _fmt(audit["top5_retained"]["sd"])),
             "- GT `[50,200]` (N={0}): `{1} ± {2}`;".format(
                 audit["gt_50_200"]["n"], _fmt(audit["gt_50_200"]["mean"]),
                 _fmt(audit["gt_50_200"]["sd"])),
             "- пересечение (N={0}, только аудит): `{1} ± {2}`.".format(
                 audit["intersection_audit_only"]["n"],
                 _fmt(audit["intersection_audit_only"]["mean"]),
                 _fmt(audit["intersection_audit_only"]["sd"]),), "",
             "Пересечение не используется как результат. Исключение top-5% зависит от известного GT и непригодно для эксплуатационной постобработки.", ""]
    titles = {"raw": "Основной raw-анализ", "clip": "Чувствительность: clipping [20,300]",
              "reject": "Чувствительность: отказ вне [20,300]",
              "oracle_top5": "Диагностика: исключение выше P95 ошибки",
              "gt_50_200": "Диагностическая GT-подгруппа [50,200]"}
    for scenario in SCENARIOS:
        lines.extend(["## " + titles[scenario], ""] + _scenario_table(results, scenario) + [""])
    lines.extend(["Значения в таблицах — cluster-equal mean `[95% cluster-bootstrap CI]`; для каждого сценария JSON также содержит segment-weighted оценки, signed SBP/DBP/MAP, BHS-доли, фактическое покрытие, причины исключения, медиану, Q1/Q3, P90/P95/P99/P99.5, максимум и концентрацию сбоев по кластерам.", "",
                  "Для MIMIC полная segment-weighted статистика использует N=27 260. Только 92 сегмента с неоднозначным происхождением исключены из кластерных оценок.", "",
                  "## Корреляция, калибровка и согласие", ""])
    for dataset in MAIN_DATASETS:
        lines.extend(["### " + LABELS[dataset], "",
                      "| Pressure | Pearson r [CI] | Spearman ρ [CI] | R² [CI] | Slope [CI] | Intercept [CI] | BA bias [CI] | Nonparametric LoA |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        diag = results["datasets"][dataset]["diagnostics"]
        for pressure in ("sbp", "dbp"):
            corr = diag["correlations"][pressure]["cluster_level_primary"]
            ba = diag["bland_altman"][pressure]
            def cell(name):
                item = corr[name]
                return "{0} [{1}; {2}]".format(_fmt(item["estimate"]),
                                               _fmt(item["ci95"][0]), _fmt(item["ci95"][1]))
            lines.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} [{7}; {8}] | {9}; {10} |".format(
                pressure.upper(), cell("pearson_r"), cell("spearman_rho"), cell("r2"),
                cell("calibration_slope"), cell("calibration_intercept"), _fmt(ba["bias"]),
                _fmt(ba["bias_ci95"][0]), _fmt(ba["bias_ci95"][1]),
                _fmt(ba["nonparametric_limits"][0]), _fmt(ba["nonparametric_limits"][1])))
        lines.append("")
    lines.extend(["Калибровка рассчитана в фиксированных 20-мм рт. ст. GT-интервалах `[20,300]` с равным весом кластеров; на рисунке показаны только интервалы с ≥10 кластерами. Bland–Altman использует signed error, cluster-equal bias, традиционные пределы `bias ± 1,96 SD` для equal-cluster mixture и основные взвешенные P2.5/P97.5 с cluster-bootstrap CI для bias и пределов.", "",
                  "Корреляция не является точностью или согласием. Limits of agreement — популяционные пределы, не CI для отдельного пациента. BHS/AAMI-показатели не означают сертификацию устройства.", "",
                  "Holdout B/C приведены только как компактная дополнительная ablation: они не использовались для повторного выбора нормализации и не получили основные диагностические рисунки.", "",
                  "## Рисунки", ""])
    for name, path in sorted(_figure_paths().items()):
        lines.append("- `{0}` — `{1}`".format(name, _relative(path)))
    lines.extend(["", "## Целостность", "",
                  "Полный JSON содержит все оценки и bootstrap summaries: `{0}`.".format(_relative(RESULTS_PATH)), "",
                  "Внутренняя проверка сверяет входы пункта 4, provenance, вычисленные признаки, рисунки и кэши предсказаний. Контрольные значения хранятся только в игнорируемом `codes/data/` и в публичный отчёт не выводятся.", "",
                  "## Ограничение этапа", "",
                  "Расчёты пункта 5 готовы для переноса, но статья и DOCX на этом этапе не изменялись. Перенос таблиц, рисунков и формулировок относится к пункту 7.", ""])
    return "\n".join(lines)


def _notebook(results):
    summary = []
    for dataset in MAIN_DATASETS:
        raw = results["datasets"][dataset]["scenarios"]["raw"]
        summary.append("{0}: N={1}, clusters={2}, waveform MAE={3}".format(
            LABELS[dataset], raw["coverage"]["segments_retained"],
            raw["coverage"]["clusters_retained"],
            _fmt(raw["metrics"]["waveform"]["cluster_equal"]["mean"])))
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": ["# Revision point 5 — verified aggregate outputs\n",
                        "This thin notebook loads only aggregate JSON and finished figures; it performs no inference."]},
            {"cell_type": "code", "execution_count": 1, "metadata": {},
             "source": ["import json\n", "from pathlib import Path\n",
                        "root = Path.cwd().parent if Path.cwd().name == 'codes' else Path.cwd()\n",
                        "results = json.loads((root / 'codes/data/revision_v2/point5/v1/results.json').read_text(encoding='utf-8'))\n",
                        "print(results['analysis'], results['primary_scenario'])"],
             "outputs": [{"name": "stdout", "output_type": "stream",
                          "text": [results["analysis"] + " raw\n"]}]},
            {"cell_type": "code", "execution_count": 2, "metadata": {},
             "source": ["for name in ('mimic', 'abp_ppg', 'vitaldb_old', 'holdout_A'):\n",
                        "    raw = results['datasets'][name]['scenarios']['raw']\n",
                        "    print(name, raw['coverage']['segments_retained'], raw['metrics']['waveform']['cluster_equal']['mean'])"],
             "outputs": [{"name": "stdout", "output_type": "stream",
                          "text": [line + "\n" for line in summary]}]},
            {"cell_type": "code", "execution_count": 3, "metadata": {},
             "source": ["from IPython.display import Image, display\n",
                        "figure_dir = root / 'codes/figures/revision_v2/point5'\n",
                        "for path in sorted(figure_dir.glob('*.png')):\n",
                        "    print(path.name)\n", "    display(Image(filename=str(path)))"],
             "outputs": [{"name": "stdout", "output_type": "stream",
                          "text": [name + ".png\n" for name in sorted(_figure_paths())]}]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                     "language_info": {"name": "python", "version": "3.6+"}},
        "nbformat": 4, "nbformat_minor": 4,
    }


def _artifact_hashes(include_report=True):
    paths = {"source_manifest": SOURCE_MANIFEST, "results": RESULTS_PATH,
             "source_core": os.path.join(HERE, "core.py"),
             "source_cli": os.path.join(HERE, "cli.py"),
             "source_tests": os.path.join(HERE, "test_point5.py")}
    for dataset in DATASETS:
        paths["features_" + dataset] = _feature_path(dataset)
    for name, path in _figure_paths().items():
        paths["figure_" + name] = path
    if include_report:
        paths["report"] = REPORT_PATH
        paths["notebook"] = NOTEBOOK_PATH
    return dict((key, sha256_file(path)) for key, path in paths.items())


def report_command(args):
    results = read_json(RESULTS_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(_render_report(results))
    write_json(NOTEBOOK_PATH, _notebook(results))
    integrity = {"schema_version": 1, "artifacts": _artifact_hashes(True),
                 "inputs": _current_input_hashes()}
    write_json(INTEGRITY_PATH, integrity)
    print("Rendered tracked report and thin executed notebook")


def _verify_figures():
    from PIL import Image
    for name, path in _figure_paths().items():
        with Image.open(path) as picture:
            if picture.width < 1800 or picture.height < 1200:
                raise AssertionError("figure dimensions too small: " + name)
            dpi = picture.info.get("dpi", (0, 0))
            if min(dpi) < 299:
                raise AssertionError("figure DPI below 300: " + name)


def _verify_feature_invariants():
    for dataset in DATASETS:
        columns = _load_features(dataset)
        masks, _ = scenario_masks(columns)
        if not np.all(masks["clip"]):
            raise AssertionError("clip must retain every segment: " + dataset)
        if np.min(columns["clip_pred_min"]) < 20.0 or np.max(columns["clip_pred_max"]) > 300.0:
            raise AssertionError("clipped prediction outside bounds: " + dataset)
        expected_reject = ~np.asarray(columns["pred_outside_20_300"], dtype=bool)
        if not np.array_equal(masks["reject"], expected_reject):
            raise AssertionError("reject mask differs from prediction-only rule: " + dataset)
        if not np.array_equal(columns["pred_outside_20_300"].astype(bool),
                              (columns["raw_pred_min"] < 20.0) |
                              (columns["raw_pred_max"] > 300.0)):
            raise AssertionError("out-of-range feature is inconsistent: " + dataset)
    mimic = _load_features("mimic")
    if len(mimic["segment_index"]) != 27260:
        raise AssertionError("MIMIC full descriptive N must be 27260")
    if int(np.sum(~mimic["cluster_included"].astype(bool))) != 92:
        raise AssertionError("MIMIC ambiguous exclusion count must be 92")


def verify_command(args):
    _validate_point4()
    source = read_json(SOURCE_MANIFEST)
    if source["inputs"] != _current_input_hashes():
        raise AssertionError("point 5 input hashes changed")
    actual_features = dict((dataset, sha256_file(_feature_path(dataset)))
                           for dataset in DATASETS)
    if source["features"] != actual_features:
        raise AssertionError("feature-table hashes changed")
    _verify_feature_invariants()
    saved = read_json(RESULTS_PATH)
    recalculated = calculate_results()
    assert_nested_equal(saved, recalculated, "point5_results")
    _verify_figures()
    integrity = read_json(INTEGRITY_PATH)
    if integrity["inputs"] != _current_input_hashes():
        raise AssertionError("final input integrity differs")
    if integrity["artifacts"] != _artifact_hashes(True):
        raise AssertionError("point 5 artifact integrity differs")
    notebook_text = json.dumps(read_json(NOTEBOOK_PATH), ensure_ascii=False)
    report_text = open(REPORT_PATH, "r", encoding="utf-8").read()
    forbidden = ("case_id", "subject_id", "record_idx", "segment_manifest.csv",
                 "mimic_segments.csv", "cluster_id")
    if any(token in notebook_text or token in report_text for token in forbidden):
        raise AssertionError("tracked material exposes identifier machinery")
    if "Статус расчёта: **выполнен и верифицирован**" not in report_text:
        raise AssertionError("report is not marked verified")
    print("VERIFY OK: point4 hashes, sources, provenance, prediction caches, feature "
          "tables, raw equivalence, scenarios, bootstrap diagnostics, figures, report, notebook")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    for name, function in (("analyze", analyze_command), ("figures", figures_command),
                           ("report", report_command), ("verify", verify_command)):
        command = subparsers.add_parser(name)
        command.set_defaults(function=function)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "function"):
        parser.print_help()
        return 2
    return args.function(args)
