"""Verify the published article bundle without any private artifact.

The canonical machine-readable source for every number in the article is
``revision_v2/point6/article_bundle_v2.json``. The CSV and Markdown tables
under ``codes/tables/revision_v2/point6/`` are derived from it by the same
functions the analysis pipeline uses.

Rebuilding the bundle itself requires the private per-segment caches, which are
not distributed. This script checks everything that *can* be checked from the
public files alone:

1. the bundle is in its canonical serialization (sorted keys, indent 2, UTF-8);
2. every published CSV and Markdown table is exactly the derived form of the
   corresponding bundle table, character for character;
3. no published file carries a case, subject, record or cluster identifier;
4. the declared analysis invariants (seed, bootstrap replicates, primary
   estimand and scenario) are the ones the article reports.

Run it from the repository root::

    python verify_public_bundle.py
"""

import sys
from pathlib import Path

from revision_v2.point6.core import (csv_bytes, find_forbidden_keys,
                                     markdown_bytes, pretty_json_bytes, read_json)

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "revision_v2" / "point6" / "article_bundle_v2.json"
TABLE_ROOT = ROOT / "codes" / "tables" / "revision_v2" / "point6"

EXPECTED = {
    "schema_version": 2,
    "seed": 20260908,
    "bootstrap_replicates": 10000,
    "primary_estimand": "mean within cluster, then equal weight per cluster",
    "primary_scenario": "raw",
    "units": "mmHg unless explicitly percent",
    "full_verification": "passed",
}
FORBIDDEN_TOKENS = ("case_id", "subject_id", "record_id", "record_idx", "cluster_id")


def lf(data: bytes) -> bytes:
    """Normalize line endings before comparing.

    The bundle and the derived tables are written with LF, but git rewrites LF
    to CRLF on checkout when ``core.autocrlf`` is enabled, which is the default
    on Windows. Without this the comparison would report every table as
    differing on a platform difference that carries no content.
    """
    return data.replace(b"\r\n", b"\n")


def main() -> int:
    failures: list[str] = []

    if not BUNDLE_PATH.is_file():
        print("missing bundle: " + str(BUNDLE_PATH))
        return 2
    bundle = read_json(BUNDLE_PATH)

    if lf(BUNDLE_PATH.read_bytes()) != lf(pretty_json_bytes(bundle)):
        failures.append("bundle is not in canonical serialization")

    for key, value in EXPECTED.items():
        if bundle.get(key) != value:
            failures.append("bundle.{0} = {1!r}, expected {2!r}".format(
                key, bundle.get(key), value))

    exposed = find_forbidden_keys(bundle)
    if exposed:
        failures.append("bundle exposes identifier keys: " + ", ".join(exposed))

    checked_files = []
    for name, table in sorted(bundle["tables"].items()):
        for suffix, payload in ((".csv", csv_bytes(table["rows"], table["fields"])),
                                (".md", markdown_bytes(table["rows"], table["fields"]))):
            path = TABLE_ROOT / (name + suffix)
            if not path.is_file():
                failures.append("missing derived table: " + path.name)
                continue
            checked_files.append(path)
            if lf(path.read_bytes()) != lf(payload):
                failures.append("derived table differs from the bundle: " + path.name)

    public_text = BUNDLE_PATH.read_text(encoding="utf-8") + "".join(
        path.read_text(encoding="utf-8") for path in checked_files)
    for token in FORBIDDEN_TOKENS:
        if token in public_text:
            failures.append("public artifact exposes identifier field: " + token)

    print("bundle : {0}".format(BUNDLE_PATH.relative_to(ROOT).as_posix()))
    print("tables : {0} files in {1}".format(
        len(checked_files), TABLE_ROOT.relative_to(ROOT).as_posix()))
    for name, table in sorted(bundle["tables"].items()):
        print("  {0:<24} {1:>4} rows, {2} fields".format(
            name, len(table["rows"]), len(table["fields"])))

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print("  - " + failure)
        return 1

    print("\nOK: bundle is canonical, every published table is its exact derived form,")
    print("    declared analysis invariants match, no identifier fields are exposed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
