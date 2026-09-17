"""Pure, Python 3.6-compatible helpers for revision point 4.

This module deliberately has no VitalDB, pandas, Torch, or TensorFlow import at
module import time.  Protocol and statistical primitives can therefore be
tested in the historical Python 3.6 environment.
"""

from __future__ import division, print_function

import hashlib
import json
import math
import os
import pickle
import sys

import numpy as np


SEED = 20260908
BOOTSTRAP_REPLICATES = 10000
SEGMENT_LENGTH = 1024
STEP = 512
MAX_SEGMENTS_PER_CASE = 100
ORIGINAL_CASE_COUNT = 100
HOLDOUT_CASE_COUNT = 100
ABP_MIN = 50.0
ABP_SCALE = 149.98749589709124

NORMALIZATIONS = {
    "A": {
        "label": "old_vitaldb_global_minmax",
        "min": -16.80837555484424,
        "max": 96.87874678552862,
        "confirmatory": True,
    },
    "B": {
        "label": "mimic_meta9_direct",
        "min_ppg": 0.0,
        "max_ppg": 4.001955034213099,
        "confirmatory": False,
    },
    "C": {
        "label": "old_vitaldb_percentile_alignment",
        "p1": 0.8139579029352338,
        "p99": 66.95604975066223,
        "clip": [-0.1, 1.1],
        "confirmatory": False,
    },
}

EXPECTED_HASHES = {
    "old_vitaldb": "840b04dad53fd49d09d0a4d75d1efb85d29f80e0ac60b8c8e5c6caeadf9ab3eb",
    "meta9": "c31914b59f5cf469551805ddba30c9e8ef95506fc46b87b315c0a189b6bbdce9",
    "approximate_pth": "ce4d15e1bd3c72c8bb9aeebca13167505cff3c805960236eac0be5cc916565fe",
    "refinement_pth": "7d856199a64f6aebe72c8a3312d7593d2d0686bf0790b9750acca7fc74a71a3b",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_json(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path, value):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle_compat(path):
    """Load a NumPy 2 pickle under NumPy 1.x as used by the ppg env."""
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        sys.modules.setdefault("numpy._core.umath", np.core.umath)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def normalization_parameters_from_source(old_vitaldb_path, meta_path):
    if sha256_file(old_vitaldb_path) != EXPECTED_HASHES["old_vitaldb"]:
        raise AssertionError("old vitaldb_raw.p SHA-256 differs from the frozen source")
    if sha256_file(meta_path) != EXPECTED_HASHES["meta9"]:
        raise AssertionError("meta9.p SHA-256 differs from the frozen source")
    old = load_pickle_compat(old_vitaldb_path)
    x = np.asarray(old["X_raw"], dtype=np.float64)
    meta = load_pickle_compat(meta_path)
    measured = {
        "A_min": float(np.min(x)),
        "A_max": float(np.max(x)),
        "B_min_ppg": float(meta["min_ppg"]),
        "B_max_ppg": float(meta["max_ppg"]),
        "C_p1": float(np.percentile(x, 1)),
        "C_p99": float(np.percentile(x, 99)),
        "abp_min": float(meta["min_abp"]),
        "abp_scale": float(meta["max_abp"] - meta["min_abp"]),
    }
    expected = {
        "A_min": NORMALIZATIONS["A"]["min"],
        "A_max": NORMALIZATIONS["A"]["max"],
        "B_min_ppg": NORMALIZATIONS["B"]["min_ppg"],
        "B_max_ppg": NORMALIZATIONS["B"]["max_ppg"],
        "C_p1": NORMALIZATIONS["C"]["p1"],
        "C_p99": NORMALIZATIONS["C"]["p99"],
        "abp_min": ABP_MIN,
        "abp_scale": ABP_SCALE,
    }
    for key in sorted(expected):
        if measured[key] != expected[key]:
            raise AssertionError("frozen parameter {0}: {1!r} != {2!r}".format(
                key, measured[key], expected[key]))
    return measured


def normalize_ppg(x_raw, variant):
    """Apply only parameters frozen on the old VitalDB/MIMIC data."""
    x = np.asarray(x_raw, dtype=np.float64)
    if variant == "A":
        p = NORMALIZATIONS["A"]
        result = (x - p["min"]) / (p["max"] - p["min"])
    elif variant == "B":
        p = NORMALIZATIONS["B"]
        result = (x - p["min_ppg"]) / (p["max_ppg"] - p["min_ppg"])
    elif variant == "C":
        p = NORMALIZATIONS["C"]
        result = (x - p["p1"]) / (p["p99"] - p["p1"])
        result = np.clip(result, p["clip"][0], p["clip"][1])
    else:
        raise ValueError("unknown normalization variant: {0}".format(variant))
    return result.astype(np.float32)


def denormalize_prediction(prediction_norm):
    return np.asarray(prediction_norm, dtype=np.float64) * ABP_SCALE + ABP_MIN


def case_seed(case_id):
    material = "{0}:{1}".format(SEED, int(case_id)).encode("ascii")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def selected_window_ranks(valid_window_count, case_id, maximum=MAX_SEGMENTS_PER_CASE):
    count = min(int(valid_window_count), int(maximum))
    if count <= 0:
        return []
    rng = np.random.RandomState(case_seed(case_id))
    return sorted(int(value) for value in
                  rng.choice(int(valid_window_count), size=count, replace=False))


def is_valid_vital_window(ppg, abp, max_nan_ratio=0.1):
    ppg = np.asarray(ppg)
    abp = np.asarray(abp)
    if np.sum(np.isnan(ppg)) / float(len(ppg)) > max_nan_ratio:
        return False
    if np.sum(np.isnan(abp)) / float(len(abp)) > max_nan_ratio:
        return False
    if np.nanstd(ppg) < 1e-6 or np.nanstd(abp) < 1e-6:
        return False
    if np.nanmin(abp) < 20 or np.nanmax(abp) > 300:
        return False
    return True


def iter_valid_windows(ppg, abp):
    sample_count = min(len(ppg), len(abp))
    rank = 0
    for start in range(0, sample_count - SEGMENT_LENGTH, STEP):
        ppg_segment = np.asarray(ppg[start:start + SEGMENT_LENGTH])
        abp_segment = np.asarray(abp[start:start + SEGMENT_LENGTH])
        if not is_valid_vital_window(ppg_segment, abp_segment):
            continue
        if np.any(np.isnan(ppg_segment)):
            valid = ~np.isnan(ppg_segment)
            ppg_segment = np.interp(np.arange(SEGMENT_LENGTH),
                                    np.arange(SEGMENT_LENGTH)[valid], ppg_segment[valid])
        if np.any(np.isnan(abp_segment)):
            valid = ~np.isnan(abp_segment)
            abp_segment = np.interp(np.arange(SEGMENT_LENGTH),
                                    np.arange(SEGMENT_LENGTH)[valid], abp_segment[valid])
        yield rank, start, ppg_segment, abp_segment
        rank += 1


def select_protocol_rows(all_case_ids, subject_by_case):
    """Select the first 100 new, non-repeated subjects after the old 100 cases."""
    all_case_ids = [int(value) for value in all_case_ids]
    original = all_case_ids[:ORIGINAL_CASE_COUNT]
    excluded_subjects = set(subject_by_case.get(case_id) for case_id in original)
    excluded_subjects.discard(None)
    rows = []
    used_subjects = set()
    for candidate_ordinal, case_id in enumerate(all_case_ids[ORIGINAL_CASE_COUNT:],
                                                ORIGINAL_CASE_COUNT):
        subject_id = subject_by_case.get(case_id)
        if subject_id is None or subject_id == "" or (
                isinstance(subject_id, float) and math.isnan(subject_id)):
            continue
        subject_id = int(subject_id)
        if subject_id in excluded_subjects or subject_id in used_subjects:
            continue
        rows.append({
            "selection_ordinal": len(rows),
            "candidate_ordinal": candidate_ordinal,
            "case_id": int(case_id),
            "subject_id": subject_id,
        })
        used_subjects.add(subject_id)
        if len(rows) == HOLDOUT_CASE_COUNT:
            break
    if len(rows) != HOLDOUT_CASE_COUNT:
        raise AssertionError("only {0} eligible holdout cases".format(len(rows)))
    return rows


def validate_protocol_rows(rows, original_case_ids, original_subject_ids):
    case_ids = [int(row["case_id"]) for row in rows]
    subject_ids = [int(row["subject_id"]) for row in rows]
    checks = {
        "cases": len(case_ids) == HOLDOUT_CASE_COUNT,
        "subjects": len(subject_ids) == HOLDOUT_CASE_COUNT,
        "unique_cases": len(set(case_ids)) == HOLDOUT_CASE_COUNT,
        "unique_subjects": len(set(subject_ids)) == HOLDOUT_CASE_COUNT,
        "no_original_cases": not (set(case_ids) & set(original_case_ids)),
        "no_original_subjects": not (set(subject_ids) & set(original_subject_ids)),
    }
    if not all(checks.values()):
        raise AssertionError("invalid protocol: {0}".format(checks))
    return checks


def mimic_cluster_included(status):
    """Ambiguous source matches must not enter record-clustered inference."""
    return status in ("exact_unique", "exact_record_only")


ERROR_KEYS = ("waveform", "sbp_abs", "dbp_abs", "map_abs",
              "sbp_signed", "dbp_signed", "map_signed")
ABSOLUTE_KEYS = ("waveform", "sbp_abs", "dbp_abs", "map_abs")
SIGNED_KEYS = ("sbp_signed", "dbp_signed", "map_signed")


def segment_errors(prediction, ground_truth):
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError("prediction and ground truth must be equal 2-D arrays")
    signed_sbp = np.max(pred, axis=1) - np.max(gt, axis=1)
    signed_dbp = np.min(pred, axis=1) - np.min(gt, axis=1)
    signed_map = np.mean(pred, axis=1) - np.mean(gt, axis=1)
    return {
        "waveform": np.mean(np.abs(pred - gt), axis=1),
        "sbp_abs": np.abs(signed_sbp),
        "dbp_abs": np.abs(signed_dbp),
        "map_abs": np.abs(signed_map),
        "sbp_signed": signed_sbp,
        "dbp_signed": signed_dbp,
        "map_signed": signed_map,
    }


def group_means(values, clusters):
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if len(values) != len(clusters):
        raise ValueError("values/clusters length mismatch")
    totals = {}
    counts = {}
    for value, cluster in zip(values, clusters):
        name = str(cluster)
        totals[name] = totals.get(name, 0.0) + float(value)
        counts[name] = counts.get(name, 0) + 1
    names = sorted(totals)
    means = np.asarray([totals[name] / counts[name] for name in names], dtype=np.float64)
    return names, means


def percentile_interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_mean_ci(cluster_values, seed=SEED, replicates=BOOTSTRAP_REPLICATES):
    values = np.asarray(cluster_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("cluster_values must be a non-empty vector")
    rng = np.random.RandomState(int(seed))
    boot = np.empty(int(replicates), dtype=np.float64)
    # Chunking avoids a large (replicates x clusters) allocation for MIMIC.
    for start in range(0, int(replicates), 256):
        stop = min(start + 256, int(replicates))
        indices = rng.randint(0, len(values), size=(stop - start, len(values)))
        boot[start:stop] = np.mean(values[indices], axis=1)
    return percentile_interval(boot)


def independent_bootstrap_difference_ci(left, right, seed=SEED,
                                        replicates=BOOTSTRAP_REPLICATES):
    """Percentile CI for mean(left)-mean(right), resampling clusters."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or not len(left) or not len(right):
        raise ValueError("left and right must be non-empty vectors")
    rng = np.random.RandomState(int(seed))
    boot = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), 256):
        stop = min(start + 256, int(replicates))
        left_indices = rng.randint(0, len(left), size=(stop - start, len(left)))
        right_indices = rng.randint(0, len(right), size=(stop - start, len(right)))
        boot[start:stop] = (np.mean(left[left_indices], axis=1) -
                            np.mean(right[right_indices], axis=1))
    return percentile_interval(boot)


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in original order."""
    p_values = [float(value) for value in p_values]
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def descriptive(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "sd": float(np.std(values)),
            "n": int(len(values))}


def metric_statistics(values, clusters, seed=SEED):
    _, cluster_values = group_means(values, clusters)
    result = {
        "segment_weighted": descriptive(values),
        "cluster_equal": descriptive(cluster_values),
    }
    result["cluster_equal"]["ci95"] = bootstrap_mean_ci(cluster_values, seed=seed)
    result["cluster_equal"]["clusters"] = int(len(cluster_values))
    return result


def dataset_statistics(errors, clusters, seed=SEED):
    result = {"metrics": {}, "bhs": {}}
    columns = []
    descriptors = []
    for key in ERROR_KEYS:
        columns.append(np.asarray(errors[key], dtype=np.float64))
        descriptors.append(("metric", key, None))
    for key in ("sbp_abs", "dbp_abs", "map_abs"):
        result["bhs"][key[:3]] = {}
        for threshold in (5, 10, 15):
            columns.append((np.asarray(errors[key]) <= threshold).astype(np.float64) * 100.0)
            descriptors.append(("bhs", key[:3], "le{0}".format(threshold)))

    matrix = np.column_stack(columns)
    cluster_names = sorted(set(str(value) for value in clusters))
    cluster_index = dict((name, index) for index, name in enumerate(cluster_names))
    cluster_matrix = np.zeros((len(cluster_names), matrix.shape[1]), dtype=np.float64)
    counts = np.zeros(len(cluster_names), dtype=np.int64)
    for row_index, cluster in enumerate(clusters):
        index = cluster_index[str(cluster)]
        cluster_matrix[index] += matrix[row_index]
        counts[index] += 1
    cluster_matrix /= counts.reshape((-1, 1))

    rng = np.random.RandomState(int(seed))
    boot = np.empty((BOOTSTRAP_REPLICATES, matrix.shape[1]), dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 64):
        stop = min(start + 64, BOOTSTRAP_REPLICATES)
        indices = rng.randint(0, len(cluster_names),
                              size=(stop - start, len(cluster_names)))
        boot[start:stop] = np.mean(cluster_matrix[indices], axis=1)

    for column_index, descriptor in enumerate(descriptors):
        stat = {
            "segment_weighted": descriptive(matrix[:, column_index]),
            "cluster_equal": descriptive(cluster_matrix[:, column_index]),
        }
        stat["cluster_equal"]["ci95"] = percentile_interval(boot[:, column_index])
        stat["cluster_equal"]["clusters"] = int(len(cluster_names))
        if descriptor[0] == "metric":
            result["metrics"][descriptor[1]] = stat
        else:
            result["bhs"][descriptor[1]][descriptor[2]] = stat
    result["n_segments"] = int(len(clusters))
    result["n_clusters"] = int(len(set(str(value) for value in clusters)))
    return result


def cliffs_delta(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    total = 0.0
    # Cluster vectors are small enough for row-wise vectorized comparisons.
    for value in left:
        total += np.sum(value > right) - np.sum(value < right)
    return float(total / (len(left) * len(right)))


def assert_nested_equal(actual, expected, path="root"):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError("{0}: dictionary keys differ".format(path))
        for key in sorted(expected):
            assert_nested_equal(actual[key], expected[key], path + "." + str(key))
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError("{0}: list lengths differ".format(path))
        for index, item in enumerate(expected):
            assert_nested_equal(actual[index], item, path + "[{0}]".format(index))
    elif isinstance(expected, float):
        if not np.isclose(float(actual), expected, rtol=0.0, atol=1e-12, equal_nan=True):
            raise AssertionError("{0}: {1!r} != {2!r}".format(path, actual, expected))
    elif actual != expected:
        raise AssertionError("{0}: {1!r} != {2!r}".format(path, actual, expected))
