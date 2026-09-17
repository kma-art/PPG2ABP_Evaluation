"""Recover segment provenance for the PPG2ABP evaluation datasets.

The detailed manifests contain deidentified source identifiers and are written to
``codes/data/provenance`` (ignored by Git).  Only aggregate results are exported
to ``.context/SEGMENT_PROVENANCE.md``.

The module intentionally supports Python 3.6 so that the VitalDB reconstruction
can be run in the original ``ppg`` conda environment (NumPy 1.17/SciPy 1.4).
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import struct
import sys
import time
from collections import defaultdict

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODES = os.path.join(ROOT, "codes")
DATA = os.path.join(CODES, "data")
RAW = os.path.join(CODES, "raw_data")
OUT = os.path.join(DATA, "provenance")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
REPORT_PATH = os.path.join(ROOT, ".context", "SEGMENT_PROVENANCE.md")

MIMIC_ROWS = 127260
MIMIC_TEST_START = 100000
MIMIC_SEGMENT_LENGTH = 1250
MIMIC_SAVED_LENGTH = 1024
MIMIC_STEP = 625
ABP_PPG_FILES = 5856
ABP_PPG_SOURCE_SEGMENTS = 11708
ABP_PPG_ACCEPTED_SEGMENTS = 11706
VITALDB_SEGMENTS = 9400
VITALDB_CASES = 100


def ensure_output():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Cannot serialize {0}".format(type(value)))


def read_summary():
    if not os.path.exists(SUMMARY_PATH):
        return {"schema_version": 1}
    with open(SUMMARY_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_summary(summary):
    ensure_output()
    summary["schema_version"] = 1
    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2,
                  sort_keys=True, default=_json_default)
        handle.write("\n")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def signal_hash(ppg, abp):
    """Hash a pair without numeric conversion; byte identity is deliberate."""
    digest = hashlib.sha256()
    for array in (ppg, abp):
        contiguous = np.ascontiguousarray(array)
        dtype = contiguous.dtype.str.encode("ascii")
        digest.update(struct.pack("<I", len(dtype)))
        digest.update(dtype)
        digest.update(struct.pack("<I", contiguous.ndim))
        for size in contiguous.shape:
            digest.update(struct.pack("<Q", int(size)))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def arrays_equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)


def pair_equal(ppg_a, abp_a, ppg_b, abp_b):
    return arrays_equal(ppg_a, ppg_b) and arrays_equal(abp_a, abp_b)


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_row_count(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def load_pickle_compat(path):
    """Load NumPy 2 pickles under NumPy 1.x (needed by the ppg environment)."""
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        sys.modules.setdefault("numpy._core.umath", np.core.umath)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def classify_matches(matches):
    """Classify exact sources while refusing to invent a disputed record."""
    if not matches:
        return "unmatched"
    unique = []
    seen = set()
    for match in matches:
        key = tuple(sorted(match.items()))
        if key not in seen:
            unique.append(match)
            seen.add(key)
    if len(unique) == 1:
        return "exact_unique"
    records = set((m.get("part"), m.get("record_idx")) for m in unique)
    if records == set([(None, None)]):
        records = set((m.get("case_id"),) for m in unique)
    if len(records) == 1:
        return "exact_record_only"
    return "ambiguous"


def split_role(hdf5_index):
    if hdf5_index >= MIMIC_TEST_START:
        return "test"
    if 80000 <= hdf5_index < 90000:
        return "validation"
    return "train"


def filter_abp_ppg(ppg, abp):
    reasons = []
    if np.isnan(ppg).any() or np.isnan(abp).any():
        reasons.append("nan")
    if np.std(ppg) < 0.01:
        reasons.append("flat_ppg")
    minimum = np.min(abp)
    maximum = np.max(abp)
    if minimum < 0:
        reasons.append("abp_negative")
    if minimum < 20 or maximum > 300:
        reasons.append("abp_outside_20_300")
    return reasons


def abp_ppg_command(args):
    import scipy.io

    ensure_output()
    source_dir = args.abp_ppg_dir
    files = sorted(name for name in os.listdir(source_dir) if name.lower().endswith(".mat"))
    file_subject_ids = set()
    eval_data = load_pickle_compat(args.abp_ppg_pickle)
    meta = load_pickle_compat(args.meta)
    min_ppg = float(meta["min_ppg"])
    max_ppg = float(meta["max_ppg"])

    manifest = []
    accepted_ppg = []
    accepted_abp = []
    accepted_hashes = []
    parse_errors = 0
    source_segment_id = 0
    evaluation_index = 0

    for file_index, filename in enumerate(files):
        path = os.path.join(source_dir, filename)
        stem = os.path.splitext(filename)[0]
        match = re.match(r"^([^_]+)_(.+)$", stem)
        subject_id = match.group(1) if match else stem
        record_id = match.group(2) if match else ""
        file_subject_ids.add(subject_id)
        try:
            mat = scipy.io.loadmat(path)
            processing = mat["signal_processing"]
        except Exception as exc:
            parse_errors += 1
            continue

        for segment_index in range(processing.shape[1]):
            entry = processing[0, segment_index]
            signal = entry["signal"]
            if isinstance(signal, np.ndarray) and signal.dtype == object:
                signal = signal.flat[0]
            signal = np.asarray(signal, dtype=np.float64)
            index = np.asarray(entry["index"]).reshape(-1)
            source_start = int(index[0]) if index.size else ""
            source_end = int(index[1]) if index.size > 1 else ""
            reasons = []
            if signal.ndim != 2 or signal.shape[0] < 2 or signal.shape[1] < MIMIC_SAVED_LENGTH:
                reasons.append("invalid_shape")
                ppg = None
                abp = None
            else:
                ppg = signal[0, :MIMIC_SAVED_LENGTH]
                abp = signal[1, :MIMIC_SAVED_LENGTH]
                reasons.extend(filter_abp_ppg(ppg, abp))
            accepted = not reasons
            pair_sha = signal_hash(ppg, abp) if ppg is not None else ""
            row = {
                "source_segment_id": source_segment_id,
                "evaluation_index": evaluation_index if accepted else "",
                "filename": filename,
                "subject_id": subject_id,
                "record_id": record_id,
                "segment_index": segment_index,
                "source_start": source_start,
                "source_end": source_end,
                "signal_length": signal.shape[1] if signal.ndim == 2 else "",
                "accepted": 1 if accepted else 0,
                "exclusion_reasons": ";".join(reasons),
                "pair_sha256": pair_sha,
                "mimic_exact_status": "not_checked",
            }
            manifest.append(row)
            source_segment_id += 1
            if accepted:
                accepted_ppg.append(ppg)
                accepted_abp.append(abp)
                accepted_hashes.append(pair_sha)
                evaluation_index += 1

    accepted_ppg = np.asarray(accepted_ppg, dtype=np.float64)
    accepted_abp = np.asarray(accepted_abp, dtype=np.float64)
    regenerated_x = ((accepted_ppg - min_ppg) / (max_ppg - min_ppg)).astype(np.float32)
    regenerated_x = regenerated_x.reshape((-1, MIMIC_SAVED_LENGTH, 1))
    regenerated_y = accepted_abp.astype(np.float32).reshape((-1, MIMIC_SAVED_LENGTH, 1))
    x_equal = arrays_equal(regenerated_x, eval_data["X_test"])
    y_equal = arrays_equal(regenerated_y, eval_data["Y_test"])

    mimic_exact = 0
    mimic_hashes = set()
    if os.path.exists(args.hdf5):
        import h5py
        with h5py.File(args.hdf5, "r") as handle:
            dataset = handle["data"]
            for start in range(0, len(dataset), args.batch_size):
                batch = dataset[start:start + args.batch_size, :, :MIMIC_SAVED_LENGTH]
                for pair in batch:
                    mimic_hashes.add(signal_hash(pair[1], pair[0]))
        accepted_index = 0
        for row in manifest:
            if row["accepted"]:
                is_match = accepted_hashes[accepted_index] in mimic_hashes
                row["mimic_exact_status"] = "exact" if is_match else "none"
                mimic_exact += int(is_match)
                accepted_index += 1

    manifest_path = os.path.join(OUT, "abp_ppg_segments.csv")
    fields = ["source_segment_id", "evaluation_index", "filename", "subject_id",
              "record_id", "segment_index", "source_start", "source_end",
              "signal_length", "accepted", "exclusion_reasons", "pair_sha256",
              "mimic_exact_status"]
    write_csv(manifest_path, fields, manifest)
    excluded = [row for row in manifest if not row["accepted"] and row["source_segment_id"] != ""]
    reason_counts = defaultdict(int)
    for row in excluded:
        for reason in row["exclusion_reasons"].split(";"):
            if reason:
                reason_counts[reason] += 1
    dataset_files = [os.path.join(source_dir, name) for name in files]
    source_listing_hash = hashlib.sha256("\n".join(
        "{0},{1},{2}".format(name, os.path.getsize(os.path.join(source_dir, name)),
                             sha256_file(os.path.join(source_dir, name)))
        for name in files).encode("utf-8")).hexdigest()
    summary = read_summary()
    summary["abp_ppg"] = {
        "files": len(files), "file_parse_errors": parse_errors,
        "source_segments": source_segment_id, "accepted_segments": len(accepted_ppg),
        "excluded_segments": len(excluded), "exclusion_reason_counts": dict(reason_counts),
        "subjects": len(file_subject_ids), "segment_subjects": len(set(row["subject_id"] for row in manifest)),
        "records": len(files), "segment_records": len(set((row["subject_id"], row["record_id"])
                                                           for row in manifest)),
        "evaluation_x_byte_equal": x_equal, "evaluation_y_byte_equal": y_equal,
        "mimic_exact_pair_matches": mimic_exact,
        "mimic_comparison_denominator": len(accepted_ppg),
        "source_listing_sha256": source_listing_hash,
        "evaluation_pickle_sha256": sha256_file(args.abp_ppg_pickle),
        "manifest": os.path.relpath(manifest_path, ROOT).replace("\\", "/"),
        "manifest_rows": len(manifest), "manifest_sha256": sha256_file(manifest_path),
    }
    write_summary(summary)
    print("ABP_PPG: {0} files, {1} source segments, {2} accepted, {3} subjects".format(
        len(files), source_segment_id, len(accepted_ppg), summary["abp_ppg"]["subjects"]))
    print("Evaluation pickle byte equality: X={0}, Y={1}; MIMIC exact pairs={2}".format(
        x_equal, y_equal, mimic_exact))


def _mimic_source_matches(hdf5_path, raw_dir, batch_size):
    import h5py

    matches = defaultdict(list)
    target_hashes = defaultdict(list)
    target_duplicate_counts = {}
    with h5py.File(hdf5_path, "r") as handle:
        dataset = handle["data"]
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]
            for offset, pair in enumerate(batch):
                target_hashes[signal_hash(pair[1], pair[0])].append(start + offset)
    for indices in target_hashes.values():
        for index in indices:
            target_duplicate_counts[index] = len(indices)

    possible_windows = 0
    record_count = 0
    with h5py.File(hdf5_path, "r") as target_handle:
        target_dataset = target_handle["data"]
        for part in range(1, 5):
            path = os.path.join(raw_dir, "Part_{0}.mat".format(part))
            with h5py.File(path, "r") as handle:
                key = "Part_{0}".format(part)
                references = handle[key]
                for record_idx in range(len(references)):
                    record_count += 1
                    record = np.asarray(handle[references[record_idx][0]])
                    for episode_start in range(0, len(record) - MIMIC_SEGMENT_LENGTH, MIMIC_STEP):
                        possible_windows += 1
                        ppg = record[episode_start:episode_start + MIMIC_SEGMENT_LENGTH, 0]
                        abp = record[episode_start:episode_start + MIMIC_SEGMENT_LENGTH, 1]
                        digest = signal_hash(ppg, abp)
                        candidate_targets = target_hashes.get(digest)
                        if not candidate_targets:
                            continue
                        source = {"part": part, "record_idx": record_idx,
                                  "episode_start": episode_start,
                                  "episode_end": episode_start + MIMIC_SEGMENT_LENGTH}
                        # A digest collision must never be accepted as provenance.
                        for target_idx in candidate_targets:
                            pair = target_dataset[target_idx]
                            if pair_equal(ppg, abp, pair[1], pair[0]):
                                matches[target_idx].append(source)
    return matches, target_duplicate_counts, possible_windows, record_count


def _record_set(rows, role):
    result = set()
    for row in rows:
        if row["split_role"] != role or row["status"] not in ("exact_unique", "exact_record_only"):
            continue
        result.add((int(row["part"]), int(row["record_idx"])))
    return result


def mimic_command(args):
    import h5py

    ensure_output()
    matches, duplicate_counts, possible_windows, record_count = _mimic_source_matches(
        args.hdf5, args.raw_dir, args.batch_size)
    rows = []
    status_counts = defaultdict(int)
    with h5py.File(args.hdf5, "r") as handle:
        dataset = handle["data"]
        total_rows = len(dataset)
        for hdf5_index in range(total_rows):
            candidates = matches.get(hdf5_index, [])
            status = classify_matches(candidates)
            status_counts[status] += 1
            part = record_idx = episode_start = episode_end = ""
            if status == "exact_unique":
                chosen = candidates[0]
                part, record_idx = chosen["part"], chosen["record_idx"]
                episode_start, episode_end = chosen["episode_start"], chosen["episode_end"]
            elif status == "exact_record_only":
                chosen = candidates[0]
                part, record_idx = chosen["part"], chosen["record_idx"]
            pair = dataset[hdf5_index]
            rows.append({
                "hdf5_index": hdf5_index, "split_role": split_role(hdf5_index),
                "part": part, "record_idx": record_idx,
                "episode_start": episode_start, "episode_end": episode_end,
                "match_count": len(candidates), "status": status,
                "hdf5_duplicate_count": duplicate_counts[hdf5_index],
                "pair_sha256": signal_hash(pair[1], pair[0]),
            })

    meta = load_pickle_compat(args.meta)
    corrected = load_pickle_compat(args.corrected_test)
    min_ppg, max_ppg = float(meta["min_ppg"]), float(meta["max_ppg"])
    min_abp, max_abp = float(meta["min_abp"]), float(meta["max_abp"])
    corrected_x_equal = True
    corrected_y_equal = True
    with h5py.File(args.hdf5, "r") as handle:
        dataset = handle["data"]
        corrected_offset = 0
        for start in range(MIMIC_TEST_START, len(dataset), args.batch_size):
            batch = dataset[start:start + args.batch_size, :, :MIMIC_SAVED_LENGTH]
            expected_x = ((batch[:, 1] - min_ppg) / (max_ppg - min_ppg)).astype(np.float32)
            expected_y = ((batch[:, 0] - min_abp) / (max_abp - min_abp)).astype(np.float32)
            count = len(batch)
            actual_x = corrected["X_test"][corrected_offset:corrected_offset + count, :, 0]
            actual_y = corrected["Y_test"][corrected_offset:corrected_offset + count, :, 0]
            corrected_x_equal = corrected_x_equal and arrays_equal(expected_x, actual_x)
            corrected_y_equal = corrected_y_equal and arrays_equal(expected_y, actual_y)
            corrected_offset += count

    train = _record_set(rows, "train")
    validation = _record_set(rows, "validation")
    test = _record_set(rows, "test")
    overlaps = {
        "train_validation": len(train & validation),
        "train_test": len(train & test),
        "validation_test": len(validation & test),
        "all_three": len(train & validation & test),
    }
    manifest_path = os.path.join(OUT, "mimic_segments.csv")
    fields = ["hdf5_index", "split_role", "part", "record_idx", "episode_start",
              "episode_end", "match_count", "status", "hdf5_duplicate_count",
              "pair_sha256"]
    write_csv(manifest_path, fields, rows)
    raw_hashes = {}
    for part in range(1, 5):
        path = os.path.join(args.raw_dir, "Part_{0}.mat".format(part))
        raw_hashes["Part_{0}.mat".format(part)] = sha256_file(path)
    summary = read_summary()
    summary["mimic"] = {
        "hdf5_rows": len(rows), "raw_records": record_count,
        "possible_windows": possible_windows, "status_counts": dict(status_counts),
        "unique_records_by_role": {"train": len(train), "validation": len(validation),
                                   "test": len(test)},
        "record_overlap": overlaps,
        "overlap_eligible_rows": sum(status_counts[s] for s in ("exact_unique", "exact_record_only")),
        "corrected_test_segments": len(corrected["X_test"]),
        "corrected_test_x_byte_equal": corrected_x_equal,
        "corrected_test_y_byte_equal": corrected_y_equal,
        "control_hdf5_index": 0,
        "control_match": {key: rows[0][key] for key in
                          ("status", "part", "record_idx", "episode_start", "episode_end")},
        "hdf5_sha256": sha256_file(args.hdf5), "raw_mat_sha256": raw_hashes,
        "corrected_test_sha256": sha256_file(args.corrected_test),
        "manifest": os.path.relpath(manifest_path, ROOT).replace("\\", "/"),
        "manifest_rows": len(rows), "manifest_sha256": sha256_file(manifest_path),
    }
    write_summary(summary)
    print("MIMIC: {0} HDF5 rows matched against {1} windows; {2}".format(
        len(rows), possible_windows, dict(status_counts)))
    print("Record overlap: {0}; corrected test byte equality X={1}, Y={2}".format(
        overlaps, corrected_x_equal, corrected_y_equal))


def is_valid_vital_segment(ppg, abp, max_nan_ratio=0.1):
    if np.sum(np.isnan(ppg)) / float(len(ppg)) > max_nan_ratio:
        return False
    if np.sum(np.isnan(abp)) / float(len(abp)) > max_nan_ratio:
        return False
    if np.nanstd(ppg) < 1e-6 or np.nanstd(abp) < 1e-6:
        return False
    if np.nanmin(abp) < 20 or np.nanmax(abp) > 300:
        return False
    return True


def vital_windows(ppg, abp, segment_length=1024, step=512):
    rank = 0
    sample_count = min(len(ppg), len(abp))
    for start in range(0, sample_count - segment_length, step):
        ppg_segment = ppg[start:start + segment_length]
        abp_segment = abp[start:start + segment_length]
        if not is_valid_vital_segment(ppg_segment, abp_segment):
            continue
        if np.any(np.isnan(ppg_segment)):
            valid = ~np.isnan(ppg_segment)
            ppg_segment = np.interp(np.arange(segment_length), np.arange(segment_length)[valid],
                                    ppg_segment[valid])
        if np.any(np.isnan(abp_segment)):
            valid = ~np.isnan(abp_segment)
            abp_segment = np.interp(np.arange(segment_length), np.arange(segment_length)[valid],
                                    abp_segment[valid])
        yield rank, start, ppg_segment, abp_segment
        rank += 1


def vitaldb_command(args):
    try:
        import vitaldb
        from scipy.signal import decimate
    except ImportError as exc:
        raise RuntimeError("Run VitalDB mode in the ppg conda environment: {0}".format(exc))

    ensure_output()
    saved = load_pickle_compat(args.vital_pickle)
    target_ppg = saved["X_raw"]
    target_abp = saved["Y_raw"]
    target_hashes = defaultdict(list)
    for segment_id in range(len(target_ppg)):
        target_hashes[signal_hash(target_ppg[segment_id], target_abp[segment_id])].append(segment_id)
    matches = defaultdict(list)

    all_cases = vitaldb.find_cases(["SNUADC/PLETH", "SNUADC/ART"])
    case_ids = [int(case_id) for case_id in all_cases[:VITALDB_CASES]]
    clinical = vitaldb.load_clinical_data(case_ids)
    subject_by_case = dict((int(row["caseid"]), int(row["subjectid"]))
                           for _, row in clinical[["caseid", "subjectid"]].iterrows())
    case_rows = []
    for ordinal, case_id in enumerate(case_ids):
        data = None
        error = ""
        attempts = 0
        for attempts in range(1, 4):
            try:
                data = vitaldb.load_case(case_id, ["SNUADC/PLETH", "SNUADC/ART"], 1.0 / 500)
                if data is None or len(data) == 0:
                    raise ValueError("empty_case")
                break
            except Exception as exc:
                error = "{0}:{1}".format(type(exc).__name__, str(exc)[:160])
                data = None
                if attempts < 3:
                    time.sleep(attempts)
        valid_count = 0
        if data is not None:
            ppg = decimate(data[:, 0], 4, ftype="fir", zero_phase=True)
            abp = decimate(data[:, 1], 4, ftype="fir", zero_phase=True)
            for rank, start, ppg_segment, abp_segment in vital_windows(ppg, abp):
                valid_count += 1
                digest = signal_hash(ppg_segment, abp_segment)
                for target_idx in target_hashes.get(digest, []):
                    if pair_equal(ppg_segment, abp_segment,
                                  target_ppg[target_idx], target_abp[target_idx]):
                        matches[target_idx].append({
                            "case_id": case_id, "subject_id": subject_by_case.get(case_id, ""),
                            "valid_window_rank": rank, "start_sample": start,
                            "end_sample": start + 1024,
                        })
        case_rows.append({
            "request_ordinal": ordinal, "case_id": case_id,
            "subject_id": subject_by_case.get(case_id, ""), "load_status": "ok" if data is not None else "failed",
            "attempts": attempts, "error": error if data is None else "",
            "valid_windows": valid_count, "selected_segments": 0,
        })
        print("VitalDB case {0}/100: case {1}, status={2}, valid={3}".format(
            ordinal + 1, case_id, "ok" if data is not None else "failed", valid_count))

    segment_rows = []
    status_counts = defaultdict(int)
    saved_order_by_case = defaultdict(int)
    selected_by_case = defaultdict(int)
    for segment_id in range(len(target_ppg)):
        candidates = matches.get(segment_id, [])
        status = classify_matches(candidates)
        status_counts[status] += 1
        row = {"segment_id": segment_id, "case_id": "", "subject_id": "",
               "valid_window_rank": "", "start_sample": "", "end_sample": "",
               "start_seconds": "", "end_seconds": "", "saved_order_within_case": "",
               "match_count": len(candidates), "status": status,
               "pair_sha256": signal_hash(target_ppg[segment_id], target_abp[segment_id])}
        if status == "exact_unique":
            chosen = candidates[0]
            row.update(chosen)
            row["start_seconds"] = chosen["start_sample"] / 125.0
            row["end_seconds"] = chosen["end_sample"] / 125.0
            row["saved_order_within_case"] = saved_order_by_case[chosen["case_id"]]
            saved_order_by_case[chosen["case_id"]] += 1
            selected_by_case[chosen["case_id"]] += 1
        elif status == "exact_record_only":
            chosen = candidates[0]
            row["case_id"] = chosen["case_id"]
            row["subject_id"] = chosen["subject_id"]
            row["saved_order_within_case"] = saved_order_by_case[chosen["case_id"]]
            saved_order_by_case[chosen["case_id"]] += 1
            selected_by_case[chosen["case_id"]] += 1
        segment_rows.append(row)
    for row in case_rows:
        row["selected_segments"] = selected_by_case[row["case_id"]]

    segment_path = os.path.join(OUT, "vitaldb_segments.csv")
    case_path = os.path.join(OUT, "vitaldb_cases.csv")
    segment_fields = ["segment_id", "case_id", "subject_id", "valid_window_rank",
                      "start_sample", "end_sample", "start_seconds", "end_seconds",
                      "saved_order_within_case", "match_count", "status", "pair_sha256"]
    case_fields = ["request_ordinal", "case_id", "subject_id", "load_status", "attempts",
                   "error", "valid_windows", "selected_segments"]
    write_csv(segment_path, segment_fields, segment_rows)
    write_csv(case_path, case_fields, case_rows)
    represented_cases = set(row["case_id"] for row in segment_rows
                            if row["status"] in ("exact_unique", "exact_record_only"))
    represented_subjects = set(row["subject_id"] for row in segment_rows
                               if row["status"] in ("exact_unique", "exact_record_only"))
    summary = read_summary()
    summary["vitaldb"] = {
        "requested_cases": len(case_rows),
        "loaded_cases": sum(row["load_status"] == "ok" for row in case_rows),
        "failed_cases": sum(row["load_status"] != "ok" for row in case_rows),
        "represented_cases": len(represented_cases), "represented_subjects": len(represented_subjects),
        "saved_segments": len(segment_rows), "status_counts": dict(status_counts),
        "valid_windows_total": sum(row["valid_windows"] for row in case_rows),
        "known_case_1_first_100": all(row["case_id"] == 1 and row["status"] == "exact_unique"
                                      for row in segment_rows[:100]),
        "source_pickle_sha256": sha256_file(args.vital_pickle),
        "segments_manifest": os.path.relpath(segment_path, ROOT).replace("\\", "/"),
        "segments_manifest_rows": len(segment_rows),
        "segments_manifest_sha256": sha256_file(segment_path),
        "cases_manifest": os.path.relpath(case_path, ROOT).replace("\\", "/"),
        "cases_manifest_rows": len(case_rows), "cases_manifest_sha256": sha256_file(case_path),
    }
    write_summary(summary)
    print("VitalDB: {0}; represented cases={1}, subjects={2}, known case-1 block={3}".format(
        dict(status_counts), len(represented_cases), len(represented_subjects),
        summary["vitaldb"]["known_case_1_first_100"]))


def _status_table(counts):
    keys = ("exact_unique", "exact_record_only", "ambiguous", "unmatched")
    return ", ".join("`{0}` — {1}".format(key, counts.get(key, 0)) for key in keys)


def report_command(args):
    summary = read_summary()
    missing = [name for name in ("vitaldb", "abp_ppg", "mimic") if name not in summary]
    if missing:
        raise RuntimeError("Missing audit results: {0}".format(", ".join(missing)))
    vital = summary["vitaldb"]
    abp = summary["abp_ppg"]
    mimic = summary["mimic"]
    text = """# Восстановление принадлежности сегментов и аудит состава выборок

## Статус и границы вывода

Происхождение сегментов восстановлено детерминированным сопоставлением цифровых
отпечатков с последующей побайтовой проверкой обеих волн. Полные манифесты с
деидентифицированными идентификаторами остаются локально в
`codes/data/provenance/` и не публикуются в Git. Корреляционное и вероятностное
связывание пациентов не выполнялось.

Идентификатор записи доказывает принадлежность записи, но не всегда пациента.
В исходных MIMIC MAT-файлах отсутствует patient ID, поэтому patient-level
пересечение MIMIC с ABP_PPG остаётся **неопределённым**, в том числе при нулевом
числе точных совпадений сигналов.

## Схемы локальных манифестов

| Файл | Строк | Ключевые поля |
|---|---:|---|
| `vitaldb_segments.csv` | {vital_segments} | `segment_id`, `case_id`, `subject_id`, `valid_window_rank`, границы в отсчётах/секундах, порядок в case, статус |
| `vitaldb_cases.csv` | {vital_cases} | запрошенный `case_id`, `subject_id`, статус/попытки загрузки, число допустимых и выбранных окон |
| `abp_ppg_segments.csv` | {abp_rows} | файл, `subject_id`, `record_id`, `segment_index`, исходный `index`, решение фильтра, статус MIMIC exact-match |
| `mimic_segments.csv` | {mimic_rows} | HDF5 index, fold-9 role, `part`, `record_idx`, границы эпизода, число источников и статус |

Статусы: `exact_unique` — единственный точный источник; `exact_record_only` —
точная запись известна, позиция неоднозначна; `ambiguous` — подходят разные
записи/case; `unmatched` — точного источника не найдено. Неоднозначные строки не
получают искусственно выбранный источник и исключаются из overlap-анализа.

## VitalDB

- Запрошено случаев: **{vital_requested}**; успешно загружено: **{vital_loaded}**;
  представлено в сохранённых данных: **{vital_represented}** случаев и
  **{vital_subjects}** субъектов.
- Сохранено сегментов: **{vital_segments}**; всего допустимых окон до исходного
  случайного ограничения по 100: **{vital_valid}**.
- Результаты сопоставления: {vital_statuses}.
- Контроль первых 100 сегментов как точных окон case 1: **{case1}**.

Исходный random seed не нужен: сохранённые PPG+ABP сопоставлялись со всеми
допустимыми окнами после воспроизведения загрузки, FIR-decimation 500→125 Гц,
фильтрации и интерполяции NaN в исходной среде.

## ABP_PPG

- MAT-файлов: **{abp_files}**; исходных сегментов: **{abp_source}**; принято:
  **{abp_accepted}**; исключено: **{abp_excluded}**.
- В именах всех файлов найдено **{abp_subjects}** subject-токенов; сегменты
  извлечены для **{abp_segment_subjects}** субъектов из **{abp_segment_records}**
  записей. Два служебных `completed*.mat` не содержат `signal_processing`.
- Причины исключения (причины могут пересекаться): `{abp_reasons}`.
- Детерминированно пересобранные `X_test`/`Y_test` побайтово совпали с
  `abp_ppg_test.p`: **X={abp_x}, Y={abp_y}**.
- Точных совпадений пары PPG+ABP с MIMIC HDF5: **{abp_mimic} из {abp_denominator}**.

## MIMIC

- Сопоставлено HDF5-эпизодов: **{mimic_rows}** со **{mimic_windows}** возможными
  окнами в **{mimic_records}** исходных записях Part_1–4.
- Результаты сопоставления: {mimic_statuses}.
- Уникальных записей по fold 9: train — **{train_records}**, validation —
  **{validation_records}**, test — **{test_records}**. Знаменатель overlap-анализа:
  **{overlap_denominator}** строк с доказанной записью.
- Record-level пересечения: train∩validation — **{tv}**, train∩test — **{tt}**,
  validation∩test — **{vt}**, во всех трёх — **{all_three}**.
- Corrected test: **{corrected_n}** сегментов (HDF5 index ≥100000), первые 1024
  точки после meta9-нормализации побайтово совпали: **X={corrected_x},
  Y={corrected_y}**.
- Контрольный HDF5-сегмент 0: `{control_status}`, Part {control_part}, запись
  {control_record}, начало {control_start}, конец {control_end}.

## Воспроизведение

CLI совместим с Python 3.6. Полный проход в исходной среде VitalDB:

```powershell
conda run -n ppg python codes/segment_provenance.py all
```

Режимы `vitaldb`, `abp-ppg`, `mimic`, `report` и `verify` можно запускать
отдельно. `verify` повторно сверяет локальные контрольные значения
исходников/манифестов, размеры, агрегаты и зафиксированные результаты
побайтовой пересборки. Контрольные значения остаются в игнорируемом
`codes/data/provenance/` и в публичный отчёт не выводятся.

## Ограничение интерпретации

Для VitalDB доступны case ID и subject ID, поэтому последующий case-level и
subject-level анализ возможен. Для ABP_PPG доступны subject ID и record ID из
имён файлов и исходные границы из MAT. Для MIMIC восстановлены только Part и
record index: открытые исходники не содержат patient ID. Следовательно,
record-level leakage можно измерить, а независимость пациентов MIMIC/ABP_PPG
доказать или опровергнуть по этим данным нельзя.
""".format(
        vital_segments=vital["segments_manifest_rows"], vital_segments_sha=vital["segments_manifest_sha256"],
        vital_cases=vital["cases_manifest_rows"], vital_cases_sha=vital["cases_manifest_sha256"],
        abp_rows=abp["manifest_rows"], abp_sha=abp["manifest_sha256"],
        mimic_rows=mimic["manifest_rows"], mimic_sha=mimic["manifest_sha256"],
        vital_requested=vital["requested_cases"], vital_loaded=vital["loaded_cases"],
        vital_represented=vital["represented_cases"], vital_subjects=vital["represented_subjects"],
        vital_valid=vital["valid_windows_total"], vital_statuses=_status_table(vital["status_counts"]),
        case1=vital["known_case_1_first_100"], vital_source_sha=vital["source_pickle_sha256"],
        abp_files=abp["files"], abp_source=abp["source_segments"], abp_accepted=abp["accepted_segments"],
        abp_excluded=abp["excluded_segments"], abp_subjects=abp["subjects"],
        abp_segment_subjects=abp["segment_subjects"], abp_segment_records=abp["segment_records"],
        abp_reasons=json.dumps(abp["exclusion_reason_counts"], ensure_ascii=False, sort_keys=True),
        abp_x=abp["evaluation_x_byte_equal"], abp_y=abp["evaluation_y_byte_equal"],
        abp_mimic=abp["mimic_exact_pair_matches"], abp_denominator=abp["mimic_comparison_denominator"],
        abp_source_sha=abp["source_listing_sha256"], abp_pickle_sha=abp["evaluation_pickle_sha256"],
        mimic_windows=mimic["possible_windows"], mimic_records=mimic["raw_records"],
        mimic_statuses=_status_table(mimic["status_counts"]),
        train_records=mimic["unique_records_by_role"]["train"],
        validation_records=mimic["unique_records_by_role"]["validation"],
        test_records=mimic["unique_records_by_role"]["test"],
        overlap_denominator=mimic["overlap_eligible_rows"], tv=mimic["record_overlap"]["train_validation"],
        tt=mimic["record_overlap"]["train_test"], vt=mimic["record_overlap"]["validation_test"],
        all_three=mimic["record_overlap"]["all_three"], corrected_n=mimic["corrected_test_segments"],
        corrected_x=mimic["corrected_test_x_byte_equal"], corrected_y=mimic["corrected_test_y_byte_equal"],
        control_status=mimic["control_match"]["status"], control_part=mimic["control_match"]["part"],
        control_record=mimic["control_match"]["record_idx"],
        control_start=mimic["control_match"]["episode_start"], control_end=mimic["control_match"]["episode_end"],
        hdf5_sha=mimic["hdf5_sha256"], corrected_sha=mimic["corrected_test_sha256"])
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    summary["report"] = {"path": os.path.relpath(REPORT_PATH, ROOT).replace("\\", "/"),
                         "sha256": sha256_file(REPORT_PATH)}
    write_summary(summary)
    print("Wrote {0}".format(REPORT_PATH))


def verify_command(args):
    summary = read_summary()
    errors = []
    for section in ("vitaldb", "abp_ppg", "mimic", "report"):
        if section not in summary:
            errors.append("missing summary section {0}".format(section))
    if errors:
        raise RuntimeError("; ".join(errors))
    vital, abp, mimic = summary["vitaldb"], summary["abp_ppg"], summary["mimic"]
    checks = [
        (vital["requested_cases"] == VITALDB_CASES, "VitalDB requested cases"),
        (vital["segments_manifest_rows"] == VITALDB_SEGMENTS, "VitalDB manifest rows"),
        (vital["status_counts"].get("exact_unique", 0) == VITALDB_SEGMENTS,
         "VitalDB exact matches"),
        (vital["known_case_1_first_100"], "VitalDB case 1 control block"),
        (abp["files"] == ABP_PPG_FILES, "ABP_PPG files"),
        (abp["source_segments"] == ABP_PPG_SOURCE_SEGMENTS, "ABP_PPG source segments"),
        (abp["accepted_segments"] == ABP_PPG_ACCEPTED_SEGMENTS, "ABP_PPG accepted segments"),
        (abp["manifest_rows"] == ABP_PPG_SOURCE_SEGMENTS, "ABP_PPG manifest rows"),
        (abp["subjects"] == 1718, "ABP_PPG filename-derived subject tokens"),
        (abp["evaluation_x_byte_equal"] and abp["evaluation_y_byte_equal"], "ABP_PPG regenerated arrays"),
        (mimic["hdf5_rows"] == MIMIC_ROWS, "MIMIC manifest rows"),
        (mimic["possible_windows"] == 514910, "MIMIC possible source windows"),
        (sum(mimic["status_counts"].values()) == MIMIC_ROWS, "MIMIC status denominator"),
        (mimic["corrected_test_segments"] == MIMIC_ROWS - MIMIC_TEST_START, "MIMIC corrected test rows"),
        (mimic["corrected_test_x_byte_equal"] and mimic["corrected_test_y_byte_equal"],
         "MIMIC corrected arrays"),
    ]
    for passed, label in checks:
        if not passed:
            errors.append("failed aggregate: {0}".format(label))
    manifest_specs = [
        (vital["segments_manifest"], vital["segments_manifest_rows"], vital["segments_manifest_sha256"]),
        (vital["cases_manifest"], vital["cases_manifest_rows"], vital["cases_manifest_sha256"]),
        (abp["manifest"], abp["manifest_rows"], abp["manifest_sha256"]),
        (mimic["manifest"], mimic["manifest_rows"], mimic["manifest_sha256"]),
    ]
    for relative, rows, expected_hash in manifest_specs:
        path = os.path.join(ROOT, relative.replace("/", os.sep))
        if not os.path.exists(path):
            errors.append("missing manifest {0}".format(relative))
        else:
            if csv_row_count(path) != rows:
                errors.append("row count changed: {0}".format(relative))
            if sha256_file(path) != expected_hash:
                errors.append("hash changed: {0}".format(relative))
    source_specs = [
        (args.vital_pickle, vital["source_pickle_sha256"]),
        (args.abp_ppg_pickle, abp["evaluation_pickle_sha256"]),
        (args.hdf5, mimic["hdf5_sha256"]),
        (args.corrected_test, mimic["corrected_test_sha256"]),
    ]
    for path, expected_hash in source_specs:
        if sha256_file(path) != expected_hash:
            errors.append("source hash changed: {0}".format(path))
    for part in range(1, 5):
        name = "Part_{0}.mat".format(part)
        path = os.path.join(args.raw_dir, name)
        if sha256_file(path) != mimic["raw_mat_sha256"][name]:
            errors.append("source hash changed: {0}".format(path))
    if sha256_file(REPORT_PATH) != summary["report"]["sha256"]:
        errors.append("report hash changed")
    if errors:
        raise RuntimeError("Verification failed:\n- " + "\n- ".join(errors))
    print("Verification passed: manifests, source hashes, aggregate invariants and regenerated arrays.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("vitaldb", "abp-ppg", "mimic", "report", "verify", "all"))
    parser.add_argument("--raw-dir", default=RAW)
    parser.add_argument("--hdf5", default=os.path.join(DATA, "data.hdf5"))
    parser.add_argument("--meta", default=os.path.join(DATA, "meta9.p"))
    parser.add_argument("--corrected-test", default=os.path.join(DATA, "test_meta9_corrected.p"))
    parser.add_argument("--vital-pickle", default=os.path.join(RAW, "vitaldb_raw.p"))
    parser.add_argument("--abp-ppg-dir", default=os.path.join(RAW, "ABP_PPG"))
    parser.add_argument("--abp-ppg-pickle", default=os.path.join(DATA, "abp_ppg_test.p"))
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    commands = ("abp-ppg", "mimic", "vitaldb", "report", "verify") if args.command == "all" else (args.command,)
    functions = {"abp-ppg": abp_ppg_command, "mimic": mimic_command,
                 "vitaldb": vitaldb_command, "report": report_command,
                 "verify": verify_command}
    for command in commands:
        functions[command](args)


if __name__ == "__main__":
    main()
