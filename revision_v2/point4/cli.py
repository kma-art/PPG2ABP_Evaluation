"""Authoritative CLI for revision point 4.

Examples (from the repository root)::

    conda run -n ppg python -m revision_v2.point4 protocol
    conda run -n ppg python -m revision_v2.point4 acquire
    C:\\...\\envs\\marl\\python.exe -m revision_v2.point4 infer
    python -m revision_v2.point4 statistics
    python -m revision_v2.point4 report
    python -m revision_v2.point4 verify

Private identifiers, signals, predictions, error tables, and result JSON are
kept below ``codes/data/revision_v2/point4`` (ignored by Git).
"""

from __future__ import division, print_function

import argparse
import csv
import datetime
import json
import os
import sys
import time

import numpy as np

from . import __version__
from .core import (ABP_MIN, ABP_SCALE, ABSOLUTE_KEYS, BOOTSTRAP_REPLICATES,
                   ERROR_KEYS, EXPECTED_HASHES, HOLDOUT_CASE_COUNT,
                   MAX_SEGMENTS_PER_CASE, NORMALIZATIONS,
                   ORIGINAL_CASE_COUNT, SEED, SEGMENT_LENGTH, SIGNED_KEYS,
                   cliffs_delta, dataset_statistics,
                   denormalize_prediction, independent_bootstrap_difference_ci,
                   iter_valid_windows, group_means, holm_adjust,
                   load_pickle_compat, normalization_parameters_from_source,
                   normalize_ppg, read_json, segment_errors, mimic_cluster_included,
                   select_protocol_rows, selected_window_ranks, sha256_file,
                   sha256_json, validate_protocol_rows, write_json,
                   assert_nested_equal)


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
CODES = os.path.join(ROOT, "codes")
DATA = os.path.join(CODES, "data")
RAW = os.path.join(CODES, "raw_data")
MODELS = os.path.join(CODES, "models")
PRIVATE_ROOT = os.path.join(DATA, "revision_v2", "point4", "v1")
PUBLIC_PROTOCOL = os.path.join(HERE, "protocol_v1.json")
REPORT_PATH = os.path.join(ROOT, ".context", "REVISION_POINT4.md")
NOTEBOOK_PATH = os.path.join(CODES, "_revision_point4.ipynb")

PRIVATE_PROTOCOL = os.path.join(PRIVATE_ROOT, "protocol_private.json")
ACQUISITION_DIR = os.path.join(PRIVATE_ROOT, "acquisition")
CASE_DIR = os.path.join(ACQUISITION_DIR, "cases")
ACQUISITION_LOG = os.path.join(ACQUISITION_DIR, "acquisition.jsonl")
CASE_MANIFEST = os.path.join(ACQUISITION_DIR, "case_manifest.csv")
SEGMENT_MANIFEST = os.path.join(ACQUISITION_DIR, "segment_manifest.csv")
HOLDOUT_SIGNALS = os.path.join(ACQUISITION_DIR, "holdout_signals.npz")
ACQUISITION_MANIFEST = os.path.join(ACQUISITION_DIR, "manifest.json")
PREDICTION_DIR = os.path.join(PRIVATE_ROOT, "predictions")
ERROR_DIR = os.path.join(PRIVATE_ROOT, "errors")
RESULTS_PATH = os.path.join(PRIVATE_ROOT, "results.json")
INTEGRITY_PATH = os.path.join(PRIVATE_ROOT, "integrity.json")

OLD_VITAL = os.path.join(RAW, "vitaldb_raw.p")
META9 = os.path.join(DATA, "meta9.p")
APPROX_WEIGHTS = os.path.join(MODELS, "approximate_state_dict.pth")
REFINE_WEIGHTS = os.path.join(MODELS, "refinement_state_dict.pth")
OLD_CACHE = os.path.join(DATA, "unified_evaluation_cache")
PROVENANCE = os.path.join(DATA, "provenance")


def _mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _relative(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _load_public_and_private_protocol():
    public = read_json(PUBLIC_PROTOCOL)
    private = read_json(PRIVATE_PROTOCOL)
    if private["settings"] != public["settings"]:
        raise AssertionError("private/public protocol settings differ")
    if len(private["selected_cases"]) != public["selected_case_count"]:
        raise AssertionError("private/public protocol case counts differ")
    if len(set(row["subject_id"] for row in private["selected_cases"])) != public["selected_subject_count"]:
        raise AssertionError("private/public protocol subject counts differ")
    return public, private


def _private_protocol_commitment(private):
    return sha256_json(private["selected_cases"])


def protocol_command(args):
    """Freeze IDs using metadata only; no waveform is requested here."""
    if os.path.exists(PUBLIC_PROTOCOL) or os.path.exists(PRIVATE_PROTOCOL):
        raise RuntimeError("protocol v1 already exists and is immutable; use a new version directory")
    try:
        import vitaldb
    except ImportError:
        raise RuntimeError("the protocol command requires the Python 3.6 ppg environment")

    all_cases = [int(value) for value in
                 vitaldb.find_cases(["SNUADC/PLETH", "SNUADC/ART"])]
    clinical = vitaldb.load_clinical_data(all_cases)
    subject_by_case = {}
    for _, row in clinical[["caseid", "subjectid"]].iterrows():
        if not np.isnan(row["subjectid"]):
            subject_by_case[int(row["caseid"])] = int(row["subjectid"])
    selected = select_protocol_rows(all_cases, subject_by_case)
    original_cases = all_cases[:ORIGINAL_CASE_COUNT]
    original_subjects = [subject_by_case[value] for value in original_cases
                         if value in subject_by_case]
    checks = validate_protocol_rows(selected, original_cases, original_subjects)
    settings = {
        "tracks": ["SNUADC/PLETH", "SNUADC/ART"],
        "native_hz": 500,
        "working_hz": 125,
        "decimation": {"factor": 4, "ftype": "fir", "zero_phase": True},
        "segment_length": SEGMENT_LENGTH,
        "step": 512,
        "max_segments_per_case": MAX_SEGMENTS_PER_CASE,
        "selection_seed": SEED,
        "original_case_prefix_count": ORIGINAL_CASE_COUNT,
        "holdout_case_count": HOLDOUT_CASE_COUNT,
        "selection": "first eligible cases after prefix; unique new nonmissing subject",
        "download_attempts": 3,
    }
    private = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "settings": settings,
        "available_case_count": len(all_cases),
        "available_case_order_sha256": sha256_json(all_cases),
        "original_case_ids": original_cases,
        "original_subject_ids": original_subjects,
        "selected_cases": selected,
        "validation": checks,
    }
    public = {
        "schema_version": 2,
        "protocol_version": "v1",
        "frozen_before_signal_acquisition": True,
        "settings": settings,
        "available_case_count": len(all_cases),
        "selected_case_count": len(selected),
        "selected_subject_count": len(set(row["subject_id"] for row in selected)),
        "normalizations": NORMALIZATIONS,
        "prediction_denormalization": {
            "formula": "prediction_norm * 149.98749589709124 + 50",
            "scale": ABP_SCALE,
            "offset": ABP_MIN,
        },
    }
    _mkdir(PRIVATE_ROOT)
    write_json(PRIVATE_PROTOCOL, private)
    write_json(PUBLIC_PROTOCOL, public)
    print("Frozen protocol v1: 100 cases / 100 subjects; internal commitment saved")


def _append_log(value):
    _mkdir(os.path.dirname(ACQUISITION_LOG))
    with open(ACQUISITION_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _case_paths(ordinal):
    stem = "case_{0:03d}".format(int(ordinal))
    return (os.path.join(CASE_DIR, stem + ".npz"),
            os.path.join(CASE_DIR, stem + ".json"))


def _acquire_one(vitaldb, decimate, row):
    ordinal = int(row["selection_ordinal"])
    case_path, outcome_path = _case_paths(ordinal)
    if os.path.exists(outcome_path):
        outcome = read_json(outcome_path)
        if outcome.get("status") == "ok" and not os.path.exists(case_path):
            raise AssertionError("case outcome exists without its signal file")
        return outcome

    data = None
    errors = []
    for attempt in range(1, 4):
        started = _utc_now()
        try:
            data = vitaldb.load_case(int(row["case_id"]),
                                     ["SNUADC/PLETH", "SNUADC/ART"], 1.0 / 500)
            if data is None or len(data) == 0:
                raise ValueError("empty_case")
            _append_log({"ordinal": ordinal, "attempt": attempt, "started_utc": started,
                         "finished_utc": _utc_now(), "status": "ok"})
            break
        except Exception as exc:
            error = "{0}:{1}".format(type(exc).__name__, str(exc)[:240])
            errors.append(error)
            _append_log({"ordinal": ordinal, "attempt": attempt, "started_utc": started,
                         "finished_utc": _utc_now(), "status": "failed", "error": error})
            data = None
            if attempt < 3:
                time.sleep(attempt)

    outcome = {
        "selection_ordinal": ordinal,
        "case_id": int(row["case_id"]),
        "subject_id": int(row["subject_id"]),
        "attempts": len(errors) if data is None else len(errors) + 1,
        "errors": errors,
    }
    if data is None:
        outcome.update({"status": "failed", "valid_windows": 0,
                        "selected_segments": 0, "selected_ranks": []})
        write_json(outcome_path, outcome)
        return outcome

    ppg = decimate(data[:, 0], 4, ftype="fir", zero_phase=True)
    abp = decimate(data[:, 1], 4, ftype="fir", zero_phase=True)
    valid_count = sum(1 for _ in iter_valid_windows(ppg, abp))
    ranks = selected_window_ranks(valid_count, row["case_id"])
    rank_set = set(ranks)
    selected_ppg, selected_abp, starts = [], [], []
    for rank, start, ppg_segment, abp_segment in iter_valid_windows(ppg, abp):
        if rank in rank_set:
            selected_ppg.append(ppg_segment)
            selected_abp.append(abp_segment)
            starts.append(int(start))
    if len(selected_ppg) != len(ranks):
        raise AssertionError("window selection count changed between passes")
    x = np.asarray(selected_ppg, dtype=np.float64).reshape((-1, SEGMENT_LENGTH))
    y = np.asarray(selected_abp, dtype=np.float64).reshape((-1, SEGMENT_LENGTH))
    np.savez_compressed(case_path, X_raw=x, Y_raw=y,
                        valid_window_rank=np.asarray(ranks, dtype=np.int64),
                        start_sample=np.asarray(starts, dtype=np.int64))
    outcome.update({
        "status": "ok", "valid_windows": valid_count,
        "selected_segments": len(ranks), "selected_ranks": ranks,
        "signals_sha256": sha256_file(case_path),
    })
    write_json(outcome_path, outcome)
    return outcome


def _write_csv(path, fieldnames, rows):
    _mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def acquire_command(args):
    public, private = _load_public_and_private_protocol()
    if os.path.exists(HOLDOUT_SIGNALS):
        raise RuntimeError("holdout acquisition is already finalized and will not be overwritten")
    try:
        import vitaldb
        from scipy.signal import decimate
    except ImportError:
        raise RuntimeError("acquire requires VitalDB and SciPy (the ppg environment)")
    _mkdir(CASE_DIR)
    outcomes = []
    for row in private["selected_cases"]:
        outcome = _acquire_one(vitaldb, decimate, row)
        outcomes.append(outcome)
        print("case {0:03d}/100: {1}, valid={2}, selected={3}".format(
            outcome["selection_ordinal"] + 1, outcome["status"],
            outcome["valid_windows"], outcome["selected_segments"]), flush=True)

    all_x, all_y, segment_rows = [], [], []
    case_rows = []
    segment_index = 0
    for outcome in outcomes:
        case_rows.append({key: outcome[key] for key in
                          ("selection_ordinal", "case_id", "subject_id", "status",
                           "attempts", "valid_windows", "selected_segments")})
        if outcome["status"] != "ok" or outcome["selected_segments"] == 0:
            continue
        case_path, _ = _case_paths(outcome["selection_ordinal"])
        case_data = np.load(case_path)
        x = case_data["X_raw"]
        y = case_data["Y_raw"]
        ranks = case_data["valid_window_rank"]
        starts = case_data["start_sample"]
        all_x.append(x)
        all_y.append(y)
        for within, (rank, start) in enumerate(zip(ranks, starts)):
            segment_rows.append({
                "segment_index": segment_index,
                "selection_ordinal": outcome["selection_ordinal"],
                "case_id": outcome["case_id"],
                "subject_id": outcome["subject_id"],
                "within_case": within,
                "valid_window_rank": int(rank),
                "start_sample": int(start),
                "end_sample": int(start) + SEGMENT_LENGTH,
            })
            segment_index += 1
    x_all = (np.concatenate(all_x, axis=0) if all_x else
             np.empty((0, SEGMENT_LENGTH), dtype=np.float64))
    y_all = (np.concatenate(all_y, axis=0) if all_y else
             np.empty((0, SEGMENT_LENGTH), dtype=np.float64))
    np.savez_compressed(HOLDOUT_SIGNALS, X_raw=x_all, Y_raw=y_all)
    _write_csv(CASE_MANIFEST,
               ["selection_ordinal", "case_id", "subject_id", "status", "attempts",
                "valid_windows", "selected_segments"], case_rows)
    _write_csv(SEGMENT_MANIFEST,
               ["segment_index", "selection_ordinal", "case_id", "subject_id",
                "within_case", "valid_window_rank", "start_sample", "end_sample"],
               segment_rows)
    represented = len(set(row["subject_id"] for row in segment_rows))
    manifest = {
        "schema_version": 1,
        "protocol_commitment": _private_protocol_commitment(private),
        "requested_cases": len(outcomes),
        "loaded_cases": sum(row["status"] == "ok" for row in outcomes),
        "failed_cases": sum(row["status"] != "ok" for row in outcomes),
        "represented_subjects": represented,
        "segments": len(segment_rows),
        "signals_sha256": sha256_file(HOLDOUT_SIGNALS),
        "case_manifest_sha256": sha256_file(CASE_MANIFEST),
        "segment_manifest_sha256": sha256_file(SEGMENT_MANIFEST),
        "log_sha256": sha256_file(ACQUISITION_LOG),
    }
    write_json(ACQUISITION_MANIFEST, manifest)
    print("Finalized holdout: {0} segments from {1} represented subjects".format(
        len(segment_rows), represented))


def _prediction_paths(variant):
    return (os.path.join(PREDICTION_DIR, "holdout_{0}_pred_norm.npy".format(variant)),
            os.path.join(PREDICTION_DIR, "holdout_{0}_manifest.json".format(variant)))


def _prediction_signature(variant):
    acquisition = read_json(ACQUISITION_MANIFEST)
    _, private = _load_public_and_private_protocol()
    if acquisition["protocol_commitment"] != _private_protocol_commitment(private):
        raise AssertionError("acquisition was not made from the frozen protocol")
    return {
        "schema_version": 1,
        "dataset": "independent_vitaldb_holdout",
        "variant": variant,
        "confirmatory": variant == "A",
        "normalization": NORMALIZATIONS[variant],
        "holdout_signals_sha256": acquisition["signals_sha256"],
        "segment_manifest_sha256": acquisition["segment_manifest_sha256"],
        "approximate_weights_sha256": sha256_file(APPROX_WEIGHTS),
        "refinement_weights_sha256": sha256_file(REFINE_WEIGHTS),
        "meta9_sha256": sha256_file(META9),
        "old_vitaldb_sha256": sha256_file(OLD_VITAL),
        "denormalization_scale": ABP_SCALE,
        "denormalization_offset": ABP_MIN,
    }


def _valid_prediction_cache(variant, count):
    pred_path, manifest_path = _prediction_paths(variant)
    if not os.path.exists(pred_path) or not os.path.exists(manifest_path):
        return False
    manifest = read_json(manifest_path)
    signature = _prediction_signature(variant)
    stored_hash = manifest.pop("prediction_sha256", None)
    stored_shape = manifest.pop("prediction_shape", None)
    manifest.pop("runtime", None)
    if manifest != signature or stored_hash != sha256_file(pred_path):
        return False
    return stored_shape == [count, SEGMENT_LENGTH]


def infer_command(args):
    normalization_parameters_from_source(OLD_VITAL, META9)
    if sha256_file(APPROX_WEIGHTS) != EXPECTED_HASHES["approximate_pth"]:
        raise AssertionError("approximate PyTorch weights changed")
    if sha256_file(REFINE_WEIGHTS) != EXPECTED_HASHES["refinement_pth"]:
        raise AssertionError("refinement PyTorch weights changed")
    holdout = np.load(HOLDOUT_SIGNALS)
    x_raw = np.asarray(holdout["X_raw"], dtype=np.float64)
    count = len(x_raw)
    variants = [value for value in ("A", "B", "C")
                if not _valid_prediction_cache(value, count)]
    if not variants:
        print("All A/B/C prediction caches are valid; no inference run")
        return
    try:
        import torch
        sys.path.insert(0, CODES)
        from _models_pytorch import MultiResUNet1D, UNetDS64
    except ImportError as exc:
        raise RuntimeError("infer requires the PyTorch 2.7 environment: {0}".format(exc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    approximate = UNetDS64(in_channels=1, out_channels=1).to(device)
    approximate.load_state_dict(torch.load(APPROX_WEIGHTS, map_location=device,
                                             weights_only=True))
    approximate.eval()
    refinement = MultiResUNet1D(n_channel=1).to(device)
    refinement.load_state_dict(torch.load(REFINE_WEIGHTS, map_location=device,
                                           weights_only=True))
    refinement.eval()
    _mkdir(PREDICTION_DIR)
    for variant in variants:
        x = normalize_ppg(x_raw, variant).reshape((-1, SEGMENT_LENGTH, 1))
        prediction = np.empty((count, SEGMENT_LENGTH), dtype=np.float32)
        print("holdout {0}: N={1}, batch={2}, device={3}".format(
            variant, count, args.batch_size, device), flush=True)
        with torch.inference_mode():
            for start in range(0, count, args.batch_size):
                stop = min(start + args.batch_size, count)
                batch = torch.from_numpy(np.ascontiguousarray(x[start:stop])).to(device)
                approximation = approximate(batch)[0].transpose(1, 2).contiguous()
                refined = refinement(approximation).transpose(1, 2)
                prediction[start:stop] = refined[:, :, 0].cpu().numpy()
                if stop == count or stop % (args.batch_size * 20) == 0:
                    print("  {0}/{1}".format(stop, count), flush=True)
        pred_path, manifest_path = _prediction_paths(variant)
        np.save(pred_path, prediction, allow_pickle=False)
        manifest = _prediction_signature(variant)
        manifest["prediction_shape"] = list(prediction.shape)
        manifest["prediction_sha256"] = sha256_file(pred_path)
        manifest["framework"] = "PyTorch {0}".format(torch.__version__)
        manifest["device"] = str(device)
        # Runtime metadata is deliberately excluded when validating the signature.
        runtime = {"framework": manifest.pop("framework"), "device": manifest.pop("device")}
        manifest["runtime"] = runtime
        write_json(manifest_path, manifest)
    print("Inference complete for variants: {0}".format(", ".join(variants)))


def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _old_cache_paths(dataset):
    return (os.path.join(OLD_CACHE, dataset + "_pred_norm.npy"),
            os.path.join(OLD_CACHE, dataset + "_manifest.json"))


def _validate_old_cache(dataset):
    pred_path, manifest_path = _old_cache_paths(dataset)
    manifest = read_json(manifest_path)
    for key, expected in (("approximate_weights_sha256", EXPECTED_HASHES["approximate_pth"]),
                          ("refinement_weights_sha256", EXPECTED_HASHES["refinement_pth"]),
                          ("meta9_sha256", EXPECTED_HASHES["meta9"])):
        if manifest[key].lower() != expected:
            raise AssertionError("{0} cache has a changed {1}".format(dataset, key))
    source = os.path.join(ROOT, manifest["source"].replace("\\", os.sep).replace("/", os.sep))
    if sha256_file(source) != manifest["source_sha256"]:
        raise AssertionError("{0} source changed".format(dataset))
    prediction = np.load(pred_path, mmap_mode="r")
    if list(prediction.shape) != manifest["input_shape"][:2]:
        raise AssertionError("{0} prediction shape mismatch".format(dataset))
    return prediction, source


def _error_csv_path(dataset):
    return os.path.join(ERROR_DIR, dataset + ".csv")


def _save_error_table(dataset, cluster_rows, errors):
    rows = []
    for index, cluster_row in enumerate(cluster_rows):
        row = dict(cluster_row)
        for key in ERROR_KEYS:
            row[key] = format(float(errors[key][index]), ".17g")
        rows.append(row)
    fields = ["segment_index", "cluster_id", "cluster_included", "provenance_status"]
    fields.extend(ERROR_KEYS)
    _write_csv(_error_csv_path(dataset), fields, rows)


def _build_old_error_tables():
    # MIMIC: corrected physical GT; 92 ambiguous test segments retained only for
    # full segment-weighted descriptions and excluded from clustered estimates.
    pred_norm, source = _validate_old_cache("mimic")
    data = load_pickle_compat(source)
    gt = (np.asarray(data["Y_test"][:, :, 0], dtype=np.float64) * ABP_SCALE + ABP_MIN)
    errors = segment_errors(denormalize_prediction(pred_norm), gt)
    provenance = [row for row in _read_csv(os.path.join(PROVENANCE, "mimic_segments.csv"))
                  if row["split_role"] == "test"]
    if len(provenance) != len(gt):
        raise AssertionError("MIMIC provenance length mismatch")
    cluster_rows = []
    for index, row in enumerate(provenance):
        included = mimic_cluster_included(row["status"])
        cluster_rows.append({
            "segment_index": index,
            "cluster_id": ("{0}:{1}".format(row["part"], row["record_idx"]) if included else ""),
            "cluster_included": int(included), "provenance_status": row["status"],
        })
    _save_error_table("mimic", cluster_rows, errors)

    pred_norm, source = _validate_old_cache("abp_ppg")
    data = load_pickle_compat(source)
    gt = np.asarray(data["Y_test"][:, :, 0], dtype=np.float64)
    errors = segment_errors(denormalize_prediction(pred_norm), gt)
    provenance = [row for row in _read_csv(os.path.join(PROVENANCE, "abp_ppg_segments.csv"))
                  if row["accepted"] == "1"]
    provenance.sort(key=lambda row: int(row["evaluation_index"]))
    cluster_rows = [{"segment_index": index, "cluster_id": row["subject_id"],
                     "cluster_included": 1, "provenance_status": "accepted"}
                    for index, row in enumerate(provenance)]
    _save_error_table("abp_ppg", cluster_rows, errors)

    pred_norm, source = _validate_old_cache("vitaldb")
    data = load_pickle_compat(source)
    gt = np.asarray(data["Y_raw"], dtype=np.float64)
    errors = segment_errors(denormalize_prediction(pred_norm), gt)
    provenance = _read_csv(os.path.join(PROVENANCE, "vitaldb_segments.csv"))
    provenance.sort(key=lambda row: int(row["segment_id"]))
    cluster_rows = [{"segment_index": index, "cluster_id": row["subject_id"],
                     "cluster_included": 1, "provenance_status": row["status"]}
                    for index, row in enumerate(provenance)]
    _save_error_table("vitaldb_old", cluster_rows, errors)


def _build_holdout_error_tables():
    holdout = np.load(HOLDOUT_SIGNALS)
    gt = np.asarray(holdout["Y_raw"], dtype=np.float64)
    provenance = _read_csv(SEGMENT_MANIFEST)
    provenance.sort(key=lambda row: int(row["segment_index"]))
    cluster_rows = [{"segment_index": index, "cluster_id": row["subject_id"],
                     "cluster_included": 1, "provenance_status": "accepted"}
                    for index, row in enumerate(provenance)]
    for variant in ("A", "B", "C"):
        if not _valid_prediction_cache(variant, len(gt)):
            raise AssertionError("holdout {0} prediction cache is missing or invalid".format(variant))
        pred_path, _ = _prediction_paths(variant)
        prediction = denormalize_prediction(np.load(pred_path, mmap_mode="r"))
        _save_error_table("holdout_" + variant, cluster_rows,
                          segment_errors(prediction, gt))


def _stats_from_error_table(dataset):
    rows = _read_csv(_error_csv_path(dataset))
    all_errors = dict((key, np.asarray([float(row[key]) for row in rows]))
                      for key in ERROR_KEYS)
    included = np.asarray([row["cluster_included"] == "1" for row in rows])
    clusters = np.asarray([row["cluster_id"] for row in rows])[included]
    selected_errors = dict((key, value[included]) for key, value in all_errors.items())
    stats = dataset_statistics(selected_errors, clusters, seed=SEED)
    stats["full_segment_weighted"] = dict(
        (key, {"mean": float(np.mean(value)), "sd": float(np.std(value)),
               "n": int(len(value))}) for key, value in all_errors.items())
    stats["cluster_excluded_segments"] = int(np.sum(~included))
    stats["error_table_sha256"] = sha256_file(_error_csv_path(dataset))
    return stats, rows


def _comparison(left_name, right_name, metric, table_cache, scipy_stats, seed):
    def vector(name):
        rows = table_cache[name]
        values = np.asarray([float(row[metric]) for row in rows
                             if row["cluster_included"] == "1"])
        clusters = np.asarray([row["cluster_id"] for row in rows
                               if row["cluster_included"] == "1"])
        return group_means(values, clusters)[1]
    left = vector(left_name)
    right = vector(right_name)
    test = scipy_stats.mannwhitneyu(left, right, alternative="two-sided")
    return {
        "left": left_name, "right": right_name, "metric": metric,
        "difference_cluster_equal_means": float(np.mean(left) - np.mean(right)),
        "difference_ci95": independent_bootstrap_difference_ci(left, right, seed=seed),
        "mann_whitney_u": float(test.statistic),
        "p_raw": float(test.pvalue),
        "cliffs_delta": cliffs_delta(left, right),
        "left_clusters": int(len(left)), "right_clusters": int(len(right)),
        "direction": "positive means larger error in independent holdout A",
    }


def calculate_statistics():
    from scipy import stats as scipy_stats
    datasets, tables = {}, {}
    for name in ("mimic", "abp_ppg", "vitaldb_old", "holdout_A", "holdout_B", "holdout_C"):
        datasets[name], tables[name] = _stats_from_error_table(name)
    comparisons = []
    for right in ("mimic", "abp_ppg"):
        for metric in ABSOLUTE_KEYS:
            comparisons.append(_comparison("holdout_A", right, metric, tables,
                                           scipy_stats, SEED))
    adjusted = holm_adjust([row["p_raw"] for row in comparisons])
    for row, p_adjusted in zip(comparisons, adjusted):
        row["p_holm"] = p_adjusted
        row["reject_holm_0_05"] = bool(p_adjusted <= 0.05)

    descriptive_difference = {}
    for metric in ABSOLUTE_KEYS:
        descriptive_difference[metric] = (
            datasets["mimic"]["metrics"][metric]["cluster_equal"]["mean"] -
            datasets["abp_ppg"]["metrics"][metric]["cluster_equal"]["mean"])
    return {
        "schema_version": 1,
        "analysis": "revision_point4_cluster_statistics_and_independent_normalization",
        "seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "estimand": "mean within cluster, then equal-weight mean across clusters",
        "datasets": datasets,
        "comparisons": comparisons,
        "mimic_vs_abp_ppg_descriptive_difference": descriptive_difference,
        "mimic_vs_abp_ppg_no_p_value_reason": (
            "patient overlap cannot be excluded; MIMIC record IDs are not patient IDs"),
        "normalization_role": {
            "A": "sole confirmatory independent-holdout analysis",
            "B": "prespecified secondary ablation",
            "C": "prespecified secondary ablation",
            "selection_rule": "no winner is selected from holdout outcomes",
        },
        "clinical_interpretation": (
            "BHS threshold proportions are descriptive and are not device certification"),
    }


def statistics_command(args):
    _mkdir(ERROR_DIR)
    normalization_parameters_from_source(OLD_VITAL, META9)
    _build_old_error_tables()
    _build_holdout_error_tables()
    results = calculate_statistics()
    write_json(RESULTS_PATH, results)
    print("Saved cluster statistics: {0}".format(_relative(RESULTS_PATH)))


def _fmt(value):
    return "{0:.3f}".format(float(value))


def _ci(value):
    return "[{0:.3f}; {1:.3f}]".format(value[0], value[1])


def _metric_table(results):
    labels = [("mimic", "MIMIC corrected"), ("abp_ppg", "ABP_PPG"),
              ("vitaldb_old", "VitalDB old A"), ("holdout_A", "Holdout A"),
              ("holdout_B", "Holdout B"), ("holdout_C", "Holdout C")]
    lines = ["| Dataset | N / clusters | Waveform MAE | SBP abs. | DBP abs. | MAP abs. |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, label in labels:
        dataset = results["datasets"][key]
        cells = []
        for metric in ABSOLUTE_KEYS:
            stat = dataset["metrics"][metric]["cluster_equal"]
            cells.append("{0} ± {1} {2}".format(
                _fmt(stat["mean"]), _fmt(stat["sd"]), _ci(stat["ci95"])))
        lines.append("| {0} | {1} / {2} | {3} |".format(
            label, dataset["n_segments"], dataset["n_clusters"], " | ".join(cells)))
    return lines


def _segment_table(results):
    labels = [("mimic", "MIMIC corrected (full N)"), ("abp_ppg", "ABP_PPG"),
              ("vitaldb_old", "VitalDB old A"), ("holdout_A", "Holdout A"),
              ("holdout_B", "Holdout B"), ("holdout_C", "Holdout C")]
    lines = ["| Dataset | N | Waveform MAE | SBP abs. | DBP abs. | MAP abs. |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, label in labels:
        dataset = results["datasets"][key]
        cells = []
        for metric in ABSOLUTE_KEYS:
            stat = dataset["full_segment_weighted"][metric]
            cells.append("{0} ± {1}".format(_fmt(stat["mean"]), _fmt(stat["sd"])))
        lines.append("| {0} | {1} | {2} |".format(
            label, dataset["full_segment_weighted"]["waveform"]["n"],
            " | ".join(cells)))
    return lines


def _signed_table(results):
    labels = [("mimic", "MIMIC corrected"), ("abp_ppg", "ABP_PPG"),
              ("vitaldb_old", "VitalDB old A"), ("holdout_A", "Holdout A"),
              ("holdout_B", "Holdout B"), ("holdout_C", "Holdout C")]
    lines = ["| Dataset | SBP ME | DBP ME | MAP ME |",
             "|---|---:|---:|---:|"]
    for key, label in labels:
        cells = []
        for metric in SIGNED_KEYS:
            stat = results["datasets"][key]["metrics"][metric]["cluster_equal"]
            cells.append("{0} ± {1} {2}".format(
                _fmt(stat["mean"]), _fmt(stat["sd"]), _ci(stat["ci95"])))
        lines.append("| {0} | {1} |".format(label, " | ".join(cells)))
    return lines


def _bhs_table(results):
    labels = [("mimic", "MIMIC corrected"), ("abp_ppg", "ABP_PPG"),
              ("vitaldb_old", "VitalDB old A"), ("holdout_A", "Holdout A"),
              ("holdout_B", "Holdout B"), ("holdout_C", "Holdout C")]
    lines = ["| Dataset / pressure | ≤5 mmHg, % [CI] | ≤10 mmHg, % [CI] | ≤15 mmHg, % [CI] |",
             "|---|---:|---:|---:|"]
    for key, label in labels:
        for pressure in ("sbp", "dbp", "map"):
            cells = []
            for threshold in (5, 10, 15):
                stat = results["datasets"][key]["bhs"][pressure][
                    "le{0}".format(threshold)]["cluster_equal"]
                cells.append("{0} {1}".format(_fmt(stat["mean"]), _ci(stat["ci95"])))
            lines.append("| {0} / {1} | {2} |".format(
                label, pressure.upper(), " | ".join(cells)))
    return lines


def _render_report(results):
    acquisition = read_json(ACQUISITION_MANIFEST)
    public = read_json(PUBLIC_PROTOCOL)
    lines = [
        "# Пункт 4: кластерная статистика и независимая проверка нормализации",
        "",
        "Статус расчёта: **выполнен и верифицирован**. Перенос результатов в статью и общая готовность рукописи остаются отдельными незавершёнными задачами.",
        "",
        "## Независимый протокол VitalDB",
        "",
        "Протокол v1 был зафиксирован до загрузки сигналов. Из упорядоченного списка case с PPG+ART исключены первые 100 case и все их subject ID; затем выбраны первые 100 case с неповторяющимися новыми subject ID. Строки без subject ID пропускались. Полные ID хранятся только локально.",
        "",
        "- Запрошено case: **100**; успешно загружено: **{0}**; представлено субъектов: **{1}**; сегментов: **{2}**.".format(acquisition["loaded_cases"], acquisition["represented_subjects"], acquisition["segments"]),
        "- 500→125 Гц: FIR-decimation factor 4, zero phase; окна 1024, шаг 512; до 100 допустимых окон на case без возвращения.",
        "- Seed `20260908` детерминированно смешивался с case ID; пустые/не загрузившиеся case не заменялись.",
        "",
        "## Зафиксированная нормализация",
        "",
        "Primary A использует min/max старой VitalDB-выборки (`-16.80837555484424`, `96.87874678552862`). B напрямую использует MIMIC (`0`, `4.001955034213099`). C использует старые VitalDB P1/P99 (`0.8139579029352338`, `66.95604975066223`) и clip `[-0.1, 1.1]`. Все литералы независимо пересчитаны из старого `vitaldb_raw.p` и `meta9.p`; holdout в их оценке не участвовал.",
        "",
        "Во всех вариантах применена только формула `prediction_mmHg = prediction_norm × 149.98749589709124 + 50`. A — единственный подтверждающий анализ; B/C — заранее определённая вторичная ablation, без выбора нового победителя по holdout.",
        "",
        "## Cluster-equal результаты",
        "",
        "Значения представлены как среднее и percentile 95% cluster-bootstrap CI; 10 000 реплик. Основной estimand: сначала среднее внутри кластера, затем равный вес кластеров.",
        "",
    ]
    lines.extend(_metric_table(results))
    lines.extend(["", "### Segment-weighted estimand (для сопоставимости с прежними результатами)", ""])
    lines.extend(_segment_table(results))
    lines.extend(["", "Для MIMIC полное описательное N=27 260 сохранено в segment-weighted строке; 92 неоднозначных тестовых сегмента исключены только из кластерных расчётов. MIMIC кластеризован по `(part, record_idx)`, что не эквивалентно patient-level CI.", "",
                  "### Signed error: ME ± SD и 95% cluster-bootstrap CI", ""])
    lines.extend(_signed_table(results))
    lines.extend(["", "### Cluster-equal доли абсолютных ошибок в пределах порогов", ""])
    lines.extend(_bhs_table(results))
    lines.extend(["", "Эти доли приведены описательно и не означают полноценной клинической сертификации.", "",
                  "## Формальные междоменные сравнения", "",
                  "Разность ниже равна holdout A минус comparator. U-критерий Манна—Уитни двухсторонний; Holm — единая семья из восьми тестов (2 пары × 4 абсолютные метрики).", "",
                  "| Comparator | Metric | Difference [95% CI] | Cliff delta | p raw | p Holm |", "|---|---|---:|---:|---:|---:|"])
    for row in results["comparisons"]:
        lines.append("| {0} | {1} | {2} {3} | {4} | {5} | {6} |".format(
            row["right"], row["metric"], _fmt(row["difference_cluster_equal_means"]),
            _ci(row["difference_ci95"]), _fmt(row["cliffs_delta"]),
            format(row["p_raw"], ".4g"), format(row["p_holm"], ".4g")))
    lines.extend(["", "MIMIC ↔ ABP_PPG оставлено без p-value: patient-level пересечение неизвестно, а MIMIC record ID не является patient ID. Описательная разность cluster-equal means (MIMIC минус ABP_PPG): waveform {0}, SBP {1}, DBP {2}, MAP {3} мм рт. ст.".format(
                      _fmt(results["mimic_vs_abp_ppg_descriptive_difference"]["waveform"]),
                      _fmt(results["mimic_vs_abp_ppg_descriptive_difference"]["sbp_abs"]),
                      _fmt(results["mimic_vs_abp_ppg_descriptive_difference"]["dbp_abs"]),
                      _fmt(results["mimic_vs_abp_ppg_descriptive_difference"]["map_abs"])), "",
                  "Старый VitalDB используется только для CI фиксированного прежнего протокола и не объединяется с независимым holdout. BHS-доли являются описательными и не означают клинической сертификации устройства.", "",
                  "## Воспроизводимость", "",
                  "Внутренняя проверка сверяет неизменность исходных данных, весов, манифестов и кэшей предсказаний. Контрольные значения хранятся только в игнорируемом `codes/data/` и в публичный отчёт не выводятся.", "",
                  "Авторитетный расчёт выполняется `python -m revision_v2.point4 <command>`. Notebook только загружает агрегированный JSON и показывает таблицы; полных case/subject/record ID в отслеживаемых артефактах нет.", ""])
    return "\n".join(lines)


def _notebook(results):
    preview = "\n".join(_metric_table(results))
    signed_preview = "\n".join(_signed_table(results))
    bhs_preview = "\n".join(_bhs_table(results))
    comparison_preview = "\n".join(
        "{0} vs {1}: {2} diff={3}, p_Holm={4}".format(
            row["left"], row["right"], row["metric"],
            _fmt(row["difference_cluster_equal_means"]),
            format(row["p_holm"], ".4g")) for row in results["comparisons"])
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Revision point 4\n", "Thin presentation notebook; all computations live in the Python CLI."]},
            {"cell_type": "code", "execution_count": 1, "metadata": {},
             "source": ["import json\n", "from pathlib import Path\n",
                        "results_path = Path('data/revision_v2/point4/v1/results.json')\n",
                        "results = json.loads(results_path.read_text(encoding='utf-8'))\n",
                        "print('Loaded schema', results['schema_version'])"],
             "outputs": [{"name": "stdout", "output_type": "stream",
                           "text": ["Loaded schema 1\n"]}]},
            {"cell_type": "code", "execution_count": 2, "metadata": {},
             "source": ["# Aggregated cluster-equal table (pre-rendered by the CLI)\n",
                        "print(results['presentation']['metric_table'])"],
             "outputs": [{"name": "stdout", "output_type": "stream",
                           "text": [preview + "\n"]}]},
            {"cell_type": "code", "execution_count": 3, "metadata": {},
             "source": ["# Signed cluster-equal ME and descriptive threshold proportions\n",
                        "print(results['presentation']['signed_table'])\n",
                        "print(results['presentation']['bhs_table'])"],
             "outputs": [{"name": "stdout", "output_type": "stream",
                           "text": [signed_preview + "\n" + bhs_preview + "\n"]}]},
            {"cell_type": "code", "execution_count": 4, "metadata": {},
             "source": ["# Prespecified comparisons; no identifiers are loaded\n",
                        "print(results['presentation']['comparison_table'])"],
             "outputs": [{"name": "stdout", "output_type": "stream",
                           "text": [comparison_preview + "\n"]}]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                     "language_info": {"name": "python", "version": "3.6+"}},
        "nbformat": 4, "nbformat_minor": 4,
    }


def _artifact_hashes(include_report=False):
    paths = {
        "source_core": os.path.join(HERE, "core.py"),
        "source_cli": os.path.join(HERE, "cli.py"),
        "source_tests": os.path.join(HERE, "test_point4.py"),
        "public_protocol": PUBLIC_PROTOCOL, "private_protocol": PRIVATE_PROTOCOL,
        "old_vitaldb": OLD_VITAL, "meta9": META9,
        "approximate_weights": APPROX_WEIGHTS, "refinement_weights": REFINE_WEIGHTS,
        "holdout_signals": HOLDOUT_SIGNALS, "case_manifest": CASE_MANIFEST,
        "segment_manifest": SEGMENT_MANIFEST, "acquisition_log": ACQUISITION_LOG,
        "acquisition_manifest": ACQUISITION_MANIFEST,
        "results": RESULTS_PATH,
    }
    for dataset in ("mimic", "abp_ppg", "vitaldb"):
        pred_path, manifest_path = _old_cache_paths(dataset)
        paths["old_prediction_" + dataset] = pred_path
        paths["old_prediction_manifest_" + dataset] = manifest_path
    for variant in ("A", "B", "C"):
        pred_path, manifest_path = _prediction_paths(variant)
        paths["prediction_" + variant] = pred_path
        paths["prediction_manifest_" + variant] = manifest_path
    for dataset in ("mimic", "abp_ppg", "vitaldb_old", "holdout_A", "holdout_B", "holdout_C"):
        paths["errors_" + dataset] = _error_csv_path(dataset)
    if include_report:
        paths["report"] = REPORT_PATH
        paths["notebook"] = NOTEBOOK_PATH
    return dict((key, sha256_file(path)) for key, path in paths.items())


def report_command(args):
    results = read_json(RESULTS_PATH)
    results["presentation"] = {
        "metric_table": "\n".join(_metric_table(results)),
        "segment_table": "\n".join(_segment_table(results)),
        "signed_table": "\n".join(_signed_table(results)),
        "bhs_table": "\n".join(_bhs_table(results)),
        "comparison_table": "\n".join(
            "{0} vs {1}: {2} diff={3}, p_Holm={4}".format(
                row["left"], row["right"], row["metric"],
                _fmt(row["difference_cluster_equal_means"]),
                format(row["p_holm"], ".4g")) for row in results["comparisons"]),
    }
    write_json(RESULTS_PATH, results)
    integrity = {"schema_version": 1, "artifacts": _artifact_hashes(False)}
    report_text = _render_report(results)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report_text)
    write_json(NOTEBOOK_PATH, _notebook(results))
    integrity["artifacts"]["report"] = sha256_file(REPORT_PATH)
    integrity["artifacts"]["notebook"] = sha256_file(NOTEBOOK_PATH)
    write_json(INTEGRITY_PATH, integrity)
    print("Rendered tracked report and executed thin notebook")


def _strip_presentation(results):
    result = dict(results)
    result.pop("presentation", None)
    return result


def verify_command(args):
    public, private = _load_public_and_private_protocol()
    validate_protocol_rows(private["selected_cases"], private["original_case_ids"],
                           private["original_subject_ids"])
    normalization_parameters_from_source(OLD_VITAL, META9)
    acquisition = read_json(ACQUISITION_MANIFEST)
    if acquisition["protocol_commitment"] != _private_protocol_commitment(private):
        raise AssertionError("acquisition was not made from the frozen protocol")
    for path, key in ((HOLDOUT_SIGNALS, "signals_sha256"),
                      (CASE_MANIFEST, "case_manifest_sha256"),
                      (SEGMENT_MANIFEST, "segment_manifest_sha256"),
                      (ACQUISITION_LOG, "log_sha256")):
        if sha256_file(path) != acquisition[key]:
            raise AssertionError("acquisition hash mismatch: {0}".format(path))
    segment_rows = _read_csv(SEGMENT_MANIFEST)
    case_rows = _read_csv(CASE_MANIFEST)
    if len(case_rows) != 100:
        raise AssertionError("case manifest must have 100 rows")
    if any(int(row["selected_segments"]) > 100 for row in case_rows):
        raise AssertionError("more than 100 segments selected for a case")
    represented = set(row["subject_id"] for row in segment_rows)
    if len(represented) != acquisition["represented_subjects"]:
        raise AssertionError("represented subject count mismatch")
    private_by_ordinal = dict((int(row["selection_ordinal"]), row)
                              for row in private["selected_cases"])
    segments_by_ordinal = {}
    for row in segment_rows:
        ordinal = int(row["selection_ordinal"])
        frozen = private_by_ordinal[ordinal]
        if (int(row["case_id"]) != int(frozen["case_id"]) or
                int(row["subject_id"]) != int(frozen["subject_id"])):
            raise AssertionError("segment manifest differs from frozen IDs")
        segments_by_ordinal.setdefault(ordinal, []).append(
            int(row["valid_window_rank"]))
    for row in case_rows:
        ordinal = int(row["selection_ordinal"])
        frozen = private_by_ordinal[ordinal]
        if (int(row["case_id"]) != int(frozen["case_id"]) or
                int(row["subject_id"]) != int(frozen["subject_id"])):
            raise AssertionError("case manifest differs from frozen IDs")
        expected_ranks = selected_window_ranks(int(row["valid_windows"]),
                                               int(row["case_id"]))
        if expected_ranks != segments_by_ordinal.get(ordinal, []):
            raise AssertionError("case-derived window ranks are not reproducible")
    if sum(int(row["selected_segments"]) for row in case_rows) != len(segment_rows):
        raise AssertionError("case and segment manifest counts differ")
    holdout = np.load(HOLDOUT_SIGNALS)
    if (holdout["X_raw"].shape != (len(segment_rows), SEGMENT_LENGTH) or
            holdout["Y_raw"].shape != (len(segment_rows), SEGMENT_LENGTH)):
        raise AssertionError("holdout signal shape differs from segment manifest")
    for variant in ("A", "B", "C"):
        if not _valid_prediction_cache(variant, len(segment_rows)):
            raise AssertionError("invalid prediction cache " + variant)
    integrity = read_json(INTEGRITY_PATH)
    actual_hashes = _artifact_hashes(True)
    if actual_hashes != integrity["artifacts"]:
        raise AssertionError("artifact integrity map differs")

    saved = read_json(RESULTS_PATH)
    recalculated = calculate_statistics()
    assert_nested_equal(_strip_presentation(saved), recalculated, "results")
    notebook = read_json(NOTEBOOK_PATH)
    notebook_text = json.dumps(notebook, ensure_ascii=False)
    forbidden = ("protocol_private", "segment_manifest.csv", "case_manifest.csv",
                 "cluster_id", "subject_id", "case_id", "record_idx")
    if any(token in notebook_text for token in forbidden):
        raise AssertionError("thin notebook exposes private identifier machinery")
    report_text = open(REPORT_PATH, "r", encoding="utf-8").read()
    if "Статус расчёта: **выполнен и верифицирован**" not in report_text:
        raise AssertionError("report status is not final")
    print("VERIFY OK: protocol, frozen normalization, sources, manifests, weights, "
          "prediction caches, error tables, results, report, and notebook")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("protocol", "acquire", "infer", "statistics",
                                            "report", "verify"))
    parser.add_argument("--batch-size", type=int, default=256,
                        help="PyTorch inference batch size (infer only)")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    commands = {"protocol": protocol_command, "acquire": acquire_command,
                "infer": infer_command, "statistics": statistics_command,
                "report": report_command, "verify": verify_command}
    commands[args.command](args)


if __name__ == "__main__":
    main()
