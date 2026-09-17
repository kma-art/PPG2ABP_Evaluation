"""Read-only point2 verifier that ignores only the volatile generation date."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "codes" / "_unified_evaluation.py"


def load_point2():
    specification = importlib.util.spec_from_file_location("point2_unified_evaluation", SOURCE)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load point2 module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    point2 = load_point2()
    verification = point2.verify_corrected()
    saved = json.loads((point2.CACHE_DIR / "metrics.json").read_text(encoding="utf-8"))
    recomputed, statuses = point2.evaluate_all(batch_size=256, force_inference=False)
    saved_comparable = dict(saved)
    recomputed_comparable = dict(recomputed)
    saved_date = saved_comparable.pop("generated")
    recomputed_date = recomputed_comparable.pop("generated")
    point2.assert_nested_close(recomputed_comparable, saved_comparable)
    expected_report = point2.render_report(saved, verification)
    if point2.REPORT_PATH.read_text(encoding="utf-8") != expected_report:
        raise AssertionError("point2 report differs from its saved dated result")
    if set(statuses.values()) != {"cache"}:
        raise AssertionError("point2 adapter unexpectedly ran inference: {0}".format(statuses))
    print(json.dumps({
        "cache_status": statuses,
        "metrics_match_excluding_generated_date": True,
        "report_matches_saved_result": True,
        "saved_generated": saved_date,
        "verification_generated": recomputed_date,
        "volatile_difference": "generated date only",
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
