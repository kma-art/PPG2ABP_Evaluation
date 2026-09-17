"""Deterministic serialization and validation helpers for point 6."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_PUBLIC_KEYS = {
    "case_id", "subject_id", "record_id", "record_idx", "cluster_id",
    "protocol_private", "segment_manifest", "case_manifest", "sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n",
                            extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: scalar_text(row.get(field)) for field in fields})
    return buffer.getvalue().encode("utf-8")


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite public value")
        return format(value, ".17g")
    return str(value)


def markdown_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    def escaped(value: Any) -> str:
        return scalar_text(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(fields) + " |",
             "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(escaped(row.get(field)) for field in fields) + " |")
    return ("\n".join(lines) + "\n").encode("utf-8")


def require_keys(value: dict[str, Any], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise AssertionError("{0} missing keys: {1}".format(label, ", ".join(missing)))


def assert_close(actual: Any, expected: Any, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError("{0}: {1!r} != {2!r}".format(label, actual, expected))


def find_forbidden_keys(value: Any, path: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_PUBLIC_KEYS:
                failures.append(path + "." + str(key))
            failures.extend(find_forbidden_keys(child, path + "." + str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(find_forbidden_keys(child, path + "[{0}]".format(index)))
    return failures


def write_if_changed(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
