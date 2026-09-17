"""Unified physical-scale evaluation for MIMIC, ABP_PPG, and VitalDB.

Large datasets and prediction caches live below ``codes/data`` and are ignored
by Git.  The only tracked output is ``.context/EVALUATION_RESULTS.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import date
from pathlib import Path
from typing import Any

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"
RAW_DIR = HERE / "raw_data"
MODEL_DIR = HERE / "models"
CACHE_DIR = DATA_DIR / "unified_evaluation_cache"
CORRECTED_PATH = DATA_DIR / "test_meta9_corrected.p"
REPORT_PATH = ROOT / ".context" / "EVALUATION_RESULTS.md"

SEGMENT_LENGTH = 1024
MIMIC_START = 100_000
MIMIC_STOP = 127_260
EXPECTED_MIMIC_N = MIMIC_STOP - MIMIC_START
ORIGINAL_TEST_ABP_RANGE = 149.9479008990615
PUBLICATION_ERROR_SCALE = 200.0

DATASETS = ("mimic", "abp_ppg", "vitaldb")
PAPER_MEANS = {
    "waveform": 4.604,
    "sbp": 5.727,
    "dbp": 3.449,
    "map": 2.310,
}


def load_meta() -> dict[str, float]:
    with (DATA_DIR / "meta9.p").open("rb") as stream:
        meta = pickle.load(stream)
    required = {"min_ppg", "max_ppg", "min_abp", "max_abp"}
    if set(meta) != required:
        raise ValueError(f"Unexpected meta9.p keys: {sorted(meta)}")
    return {key: float(value) for key, value in meta.items()}


def abp_scale(meta: dict[str, float]) -> float:
    """The sole denormalization coefficient used by the main comparison."""
    return meta["max_abp"] - meta["min_abp"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_corrected(force: bool = False) -> dict[str, Any]:
    """Build MIMIC test data directly from HDF5 using fold-9 metadata."""
    meta = load_meta()
    if force or not CORRECTED_PATH.exists():
        x_test = np.empty((EXPECTED_MIMIC_N, SEGMENT_LENGTH, 1), dtype=np.float32)
        y_test = np.empty_like(x_test)
        ppg_scale = meta["max_ppg"] - meta["min_ppg"]
        target_scale = abp_scale(meta)

        with h5py.File(DATA_DIR / "data.hdf5", "r") as h5:
            data = h5["data"]
            if len(data) != MIMIC_STOP:
                raise ValueError(f"Unexpected HDF5 length: {len(data)}")
            for start in range(MIMIC_START, MIMIC_STOP, 512):
                stop = min(start + 512, MIMIC_STOP)
                raw = np.asarray(data[start:stop, :, :SEGMENT_LENGTH])
                out_start = start - MIMIC_START
                out_stop = stop - MIMIC_START
                x_test[out_start:out_stop, :, 0] = (
                    (raw[:, 1, :] - meta["min_ppg"]) / ppg_scale
                ).astype(np.float32)
                y_test[out_start:out_stop, :, 0] = (
                    (raw[:, 0, :] - meta["min_abp"]) / target_scale
                ).astype(np.float32)

        payload = {
            "X_test": x_test,
            "Y_test": y_test,
            "source": "data.hdf5[100000:127260, :, :1024]",
            "normalization": "meta9.p exact min-max",
            "n_segments": EXPECTED_MIMIC_N,
        }
        with CORRECTED_PATH.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    return verify_corrected()


def verify_corrected() -> dict[str, Any]:
    """Check shape, order, and inverse transformations against source HDF5."""
    meta = load_meta()
    with CORRECTED_PATH.open("rb") as stream:
        corrected = pickle.load(stream)
    x_test = np.asarray(corrected["X_test"])
    y_test = np.asarray(corrected["Y_test"])
    expected_shape = (EXPECTED_MIMIC_N, SEGMENT_LENGTH, 1)
    if x_test.shape != expected_shape or y_test.shape != expected_shape:
        raise AssertionError(
            f"Corrected shapes {x_test.shape}/{y_test.shape}, expected {expected_shape}"
        )

    target_scale = abp_scale(meta)
    ppg_scale = meta["max_ppg"] - meta["min_ppg"]
    gt_max_error = 0.0
    ppg_max_error = 0.0
    original_gt_max_error = 0.0
    original_x_float32_equal = True
    with (DATA_DIR / "test_original.p").open("rb") as stream:
        original = pickle.load(stream)
    original_x = np.asarray(original["X_test"])
    original_y = np.asarray(original["Y_test"])

    with h5py.File(DATA_DIR / "data.hdf5", "r") as h5:
        data = h5["data"]
        for offset in range(0, EXPECTED_MIMIC_N, 512):
            stop = min(offset + 512, EXPECTED_MIMIC_N)
            raw = np.asarray(
                data[MIMIC_START + offset:MIMIC_START + stop, :, :SEGMENT_LENGTH]
            )
            sl = slice(offset, stop)
            gt_restored = (
                y_test[sl, :, 0].astype(np.float64) * target_scale + meta["min_abp"]
            )
            ppg_restored = (
                x_test[sl, :, 0].astype(np.float64) * ppg_scale + meta["min_ppg"]
            )
            original_restored = (
                original_y[sl, :, 0] * ORIGINAL_TEST_ABP_RANGE + meta["min_abp"]
            )
            gt_max_error = max(gt_max_error, float(np.max(np.abs(gt_restored - raw[:, 0, :]))))
            ppg_max_error = max(ppg_max_error, float(np.max(np.abs(ppg_restored - raw[:, 1, :]))))
            original_gt_max_error = max(
                original_gt_max_error,
                float(np.max(np.abs(original_restored - raw[:, 0, :]))),
            )
            original_x_float32_equal = original_x_float32_equal and np.array_equal(
                x_test[sl], original_x[sl].astype(np.float32)
            )

    # Float32 storage bounds the round-trip error; this is numerical, not signal error.
    if gt_max_error > 2e-5 or ppg_max_error > 5e-7:
        raise AssertionError(
            f"Corrected round-trip error too large: ABP={gt_max_error}, PPG={ppg_max_error}"
        )
    if original_gt_max_error > 1e-10:
        raise AssertionError(
            f"Original GT does not match inferred range: {original_gt_max_error}"
        )
    if not original_x_float32_equal:
        raise AssertionError("Corrected and original PPG differ after float32 conversion")

    return {
        "path": str(CORRECTED_PATH.relative_to(ROOT)),
        "shape": list(expected_shape),
        "dtype": str(x_test.dtype),
        "gt_roundtrip_max_abs": gt_max_error,
        "ppg_roundtrip_max_abs": ppg_max_error,
        "original_gt_scale": ORIGINAL_TEST_ABP_RANGE,
        "original_gt_roundtrip_max_abs": original_gt_max_error,
        "original_ppg_float32_equal": original_x_float32_equal,
    }


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, Path, str]:
    """Return normalized PPG, physical GT, source path, and preprocessing note."""
    meta = load_meta()
    if name == "mimic":
        with CORRECTED_PATH.open("rb") as stream:
            data = pickle.load(stream)
        x = np.asarray(data["X_test"], dtype=np.float32)
        gt = (
            np.asarray(data["Y_test"][:, :, 0], dtype=np.float64) * abp_scale(meta)
            + meta["min_abp"]
        )
        return x, gt, CORRECTED_PATH, "PPG and GT normalized with meta9.p"

    if name == "abp_ppg":
        source = DATA_DIR / "abp_ppg_test.p"
        with source.open("rb") as stream:
            data = pickle.load(stream)
        x = np.asarray(data["X_test"], dtype=np.float32)
        gt = np.asarray(data["Y_test"][:, :, 0], dtype=np.float64)
        return x, gt, source, "PPG normalized with meta9.p; GT already mmHg"

    if name == "vitaldb":
        source = RAW_DIR / "vitaldb_raw.p"
        with source.open("rb") as stream:
            data = pickle.load(stream)
        x_raw = np.asarray(data["X_raw"], dtype=np.float64)
        gt = np.asarray(data["Y_raw"], dtype=np.float64)
        ppg_min = float(x_raw.min())
        ppg_max = float(x_raw.max())
        x = ((x_raw - ppg_min) / (ppg_max - ppg_min)).astype(np.float32)
        x = x.reshape(-1, SEGMENT_LENGTH, 1)
        return x, gt, source, "VitalDB global min-max PPG; GT already mmHg"

    raise ValueError(f"Unknown dataset: {name}")


def cache_paths(name: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{name}_pred_norm.npy", CACHE_DIR / f"{name}_manifest.json"


def cache_signature(name: str, source: Path, shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "dataset": name,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "input_shape": list(shape),
        "approximate_weights_sha256": sha256(MODEL_DIR / "approximate_state_dict.pth"),
        "refinement_weights_sha256": sha256(MODEL_DIR / "refinement_state_dict.pth"),
        "meta9_sha256": sha256(DATA_DIR / "meta9.p"),
    }


def predict_norm(
    name: str,
    x: np.ndarray,
    source: Path,
    batch_size: int = 256,
    force: bool = False,
) -> tuple[np.ndarray, str]:
    """Run the two-stage PyTorch model or load a validated local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pred_path, manifest_path = cache_paths(name)
    signature = cache_signature(name, source, tuple(x.shape))
    if not force and pred_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest == signature:
            pred = np.load(pred_path, mmap_mode="r")
            if pred.shape == (len(x), SEGMENT_LENGTH):
                return np.asarray(pred), "cache"

    import torch

    try:
        from ._models_pytorch import MultiResUNet1D, UNetDS64
    except ImportError:
        from _models_pytorch import MultiResUNet1D, UNetDS64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    approximate = UNetDS64(in_channels=1, out_channels=1).to(device)
    approximate.load_state_dict(
        torch.load(
            MODEL_DIR / "approximate_state_dict.pth",
            map_location=device,
            weights_only=True,
        )
    )
    approximate.eval()
    refinement = MultiResUNet1D(n_channel=1).to(device)
    refinement.load_state_dict(
        torch.load(
            MODEL_DIR / "refinement_state_dict.pth",
            map_location=device,
            weights_only=True,
        )
    )
    refinement.eval()

    pred = np.empty((len(x), SEGMENT_LENGTH), dtype=np.float32)
    print(f"[{name}] inference: N={len(x)}, batch={batch_size}, device={device}", flush=True)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            batch = torch.from_numpy(np.ascontiguousarray(x[start:stop])).to(device)
            approximation = approximate(batch)[0].transpose(1, 2).contiguous()
            refined = refinement(approximation).transpose(1, 2)
            pred[start:stop] = refined[:, :, 0].cpu().numpy()
            if stop == len(x) or stop % (batch_size * 100) == 0:
                print(f"[{name}] {stop}/{len(x)}", flush=True)

    np.save(pred_path, pred, allow_pickle=False)
    manifest_path.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return pred, f"inference:{device}"


def summary(values: np.ndarray, signed: bool = False) -> dict[str, float]:
    result = {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }
    if signed:
        result["p05"] = float(np.percentile(values, 5))
    return result


def bhs(error: np.ndarray) -> dict[str, Any]:
    percentages = [float(np.mean(error <= threshold) * 100.0) for threshold in (5, 10, 15)]
    if all(value >= boundary for value, boundary in zip(percentages, (60, 85, 95))):
        grade = "A"
    elif all(value >= boundary for value, boundary in zip(percentages, (50, 75, 90))):
        grade = "B"
    elif all(value >= boundary for value, boundary in zip(percentages, (40, 65, 85))):
        grade = "C"
    else:
        grade = "D"
    return {"le5": percentages[0], "le10": percentages[1], "le15": percentages[2], "grade": grade}


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    waveform_error = np.mean(np.abs(pred - gt), axis=1)
    pred_values = {
        "sbp": np.max(pred, axis=1),
        "dbp": np.min(pred, axis=1),
        "map": np.mean(pred, axis=1),
    }
    gt_values = {
        "sbp": np.max(gt, axis=1),
        "dbp": np.min(gt, axis=1),
        "map": np.mean(gt, axis=1),
    }

    pressure: dict[str, Any] = {}
    for key in ("sbp", "dbp", "map"):
        signed_error = pred_values[key] - gt_values[key]
        absolute_error = np.abs(signed_error)
        signed_stats = summary(signed_error, signed=True)
        pressure[key] = {
            "absolute": summary(absolute_error),
            "signed": signed_stats,
            "bhs": bhs(absolute_error),
            "aami": (
                "PASS"
                if abs(signed_stats["mean"]) <= 5.0 and signed_stats["sd"] <= 8.0
                else "FAIL"
            ),
        }

    return {
        "n": int(len(gt)),
        "waveform": summary(waveform_error),
        "pressure": pressure,
        "range": {
            "gt_min": float(np.min(gt)),
            "gt_max": float(np.max(gt)),
            "pred_min": float(np.min(pred)),
            "pred_max": float(np.max(pred)),
        },
        "outliers": {
            "definition": "segment waveform MAE threshold; prediction outside [20, 300] mmHg",
            "mae_gt50": int(np.sum(waveform_error > 50.0)),
            "mae_gt100": int(np.sum(waveform_error > 100.0)),
            "mae_gt500": int(np.sum(waveform_error > 500.0)),
            "pred_outside_20_300": int(
                np.sum((np.min(pred, axis=1) < 20.0) | (np.max(pred, axis=1) > 300.0))
            ),
        },
    }


def publication_reproduction(pred_norm: np.ndarray) -> dict[str, Any]:
    """Reproduce, but explicitly do not endorse, the publication's scale bug."""
    meta = load_meta()
    with (DATA_DIR / "test_original.p").open("rb") as stream:
        original = pickle.load(stream)
    original_x = np.asarray(original["X_test"], dtype=np.float32)
    corrected_x, _, _, _ = load_dataset("mimic")
    if not np.array_equal(original_x, corrected_x):
        raise AssertionError("Publication reproduction cannot reuse the MIMIC cache: PPG differs")

    gt = np.asarray(original["Y_test"][:, :, 0], dtype=np.float64)
    gt = gt * PUBLICATION_ERROR_SCALE + meta["min_abp"]
    pred = pred_norm.astype(np.float64) * PUBLICATION_ERROR_SCALE + meta["min_abp"]
    metrics = compute_metrics(pred, gt)
    measured = {
        "waveform": metrics["waveform"]["mean"],
        "sbp": metrics["pressure"]["sbp"]["absolute"]["mean"],
        "dbp": metrics["pressure"]["dbp"]["absolute"]["mean"],
        "map": metrics["pressure"]["map"]["absolute"]["mean"],
    }
    return {
        key: {
            "paper": PAPER_MEANS[key],
            "reproduced": measured[key],
            "deviation": measured[key] - PAPER_MEANS[key],
            "absolute_deviation": abs(measured[key] - PAPER_MEANS[key]),
        }
        for key in PAPER_MEANS
    }


def evaluate_all(
    batch_size: int = 256,
    force_inference: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Compute all metrics from validated prediction caches (creating as needed)."""
    meta = load_meta()
    target_scale = abp_scale(meta)
    datasets: dict[str, Any] = {}
    cache_status: dict[str, str] = {}
    mimic_pred_norm: np.ndarray | None = None

    for name in DATASETS:
        x, gt, source, preprocessing = load_dataset(name)
        pred_norm, status = predict_norm(
            name,
            x,
            source,
            batch_size=batch_size,
            force=force_inference,
        )
        if pred_norm.shape != gt.shape:
            raise AssertionError(f"{name}: prediction/GT shape mismatch {pred_norm.shape}/{gt.shape}")
        pred = pred_norm.astype(np.float64) * target_scale + meta["min_abp"]
        metrics = compute_metrics(pred, gt)
        metrics["preprocessing"] = preprocessing
        datasets[name] = metrics
        cache_status[name] = status
        if name == "mimic":
            mimic_pred_norm = pred_norm

    assert mimic_pred_norm is not None
    results = {
        "generated": date.today().isoformat(),
        "meta": {
            **meta,
            "abp_scale": target_scale,
            "rule": "prediction_mmHg = prediction_norm * (max_abp - min_abp) + min_abp",
        },
        "datasets": datasets,
        "ratios": {
            "abp_ppg_over_mimic": (
                datasets["abp_ppg"]["waveform"]["mean"]
                / datasets["mimic"]["waveform"]["mean"]
            ),
            "vitaldb_over_mimic": (
                datasets["vitaldb"]["waveform"]["mean"]
                / datasets["mimic"]["waveform"]["mean"]
            ),
        },
        "publication_reproduction_x200": publication_reproduction(mimic_pred_norm),
    }
    return results, cache_status


LABELS = {"mimic": "MIMIC corrected", "abp_ppg": "ABP_PPG", "vitaldb": "VitalDB"}
PRESSURE_LABELS = {"sbp": "SBP", "dbp": "DBP", "map": "MAP"}


def f3(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def f6(value: float) -> str:
    return f"{value:.6f}".replace(".", ",")


def f15(value: float) -> str:
    return f"{value:.15f}".replace(".", ",")


def render_report(results: dict[str, Any], verification: dict[str, Any]) -> str:
    scale = results["meta"]["abp_scale"]
    lines = [
        "# Единая оценка PPG2ABP в физической шкале",
        "",
        f"Дата расчёта: {results['generated']}.",
        "",
        "## Основание расчёта",
        "",
        "Во всех трёх основных расчётах prediction денормализуется одной формулой",
        "из `meta9.p`:",
        "",
        "```python",
        "prediction_mmhg = prediction_norm * (max_abp - min_abp) + min_abp",
        "```",
        "",
        f"Точный коэффициент fold 9: `{scale:.14f}`; offset:",
        f"`{results['meta']['min_abp']:.1f}` мм рт. ст. Литералы 150 и 200 в",
        "основном расчёте не используются. MIMIC заново собран из",
        "`data.hdf5[100000:127260]`; ABP_PPG и VitalDB сохраняют GT в исходных",
        "мм рт. ст. Для VitalDB сохранён global min–max PPG.",
        "",
        "## Waveform MAE",
        "",
        "Статистика вычислена по сегментным MAE (1024 точки на сегменте).",
        "",
        "| Набор | N | Mean | SD | Median | P95 |",
        "|-------|--:|-----:|---:|-------:|----:|",
    ]
    for name in DATASETS:
        data = results["datasets"][name]
        wave = data["waveform"]
        n_formatted = f"{data['n']:,}".replace(",", " ")
        lines.append(
            f"| {LABELS[name]} | {n_formatted} | {f3(wave['mean'])} | {f3(wave['sd'])} | "
            f"{f3(wave['median'])} | {f3(wave['p95'])} |"
        )

    lines += [
        "",
        "Все значения в таблицах метрик приведены в мм рт. ст.",
        "",
        "## Ошибки SBP, DBP и MAP",
        "",
        "### Абсолютная ошибка",
        "",
        "| Набор | Показатель | Mean | SD | Median | P95 |",
        "|-------|------------|-----:|---:|-------:|----:|",
    ]
    for name in DATASETS:
        for key in ("sbp", "dbp", "map"):
            values = results["datasets"][name]["pressure"][key]["absolute"]
            lines.append(
                f"| {LABELS[name]} | {PRESSURE_LABELS[key]} | {f3(values['mean'])} | "
                f"{f3(values['sd'])} | {f3(values['median'])} | {f3(values['p95'])} |"
            )

    lines += [
        "",
        "### Знаковая ошибка (prediction − GT)",
        "",
        "| Набор | Показатель | ME | SD | Median | P05 | P95 |",
        "|-------|------------|---:|---:|-------:|----:|----:|",
    ]
    for name in DATASETS:
        for key in ("sbp", "dbp", "map"):
            values = results["datasets"][name]["pressure"][key]["signed"]
            lines.append(
                f"| {LABELS[name]} | {PRESSURE_LABELS[key]} | {f3(values['mean'])} | "
                f"{f3(values['sd'])} | {f3(values['median'])} | {f3(values['p05'])} | "
                f"{f3(values['p95'])} |"
            )

    lines += [
        "",
        "## BHS",
        "",
        "Пороговые доли рассчитаны после перевода в мм рт. ст.; критерии грейдов",
        "A/B/C: 60/85/95 %, 50/75/90 % и 40/65/85 % соответственно.",
        "",
        "| Набор | Показатель | ≤5 | ≤10 | ≤15 | Grade |",
        "|-------|------------|---:|----:|----:|:-----:|",
    ]
    for name in DATASETS:
        for key in ("sbp", "dbp", "map"):
            values = results["datasets"][name]["pressure"][key]["bhs"]
            lines.append(
                f"| {LABELS[name]} | {PRESSURE_LABELS[key]} | {f3(values['le5'])}% | "
                f"{f3(values['le10'])}% | {f3(values['le15'])}% | **{values['grade']}** |"
            )

    lines += [
        "",
        "## AAMI",
        "",
        "PASS требует одновременно |ME| ≤ 5 и SD ≤ 8 мм рт. ст. Оценка",
        "сегментная и не заменяет независимую subject-level валидацию прибора.",
        "",
        "| Набор | Показатель | ME ± SD | Результат |",
        "|-------|------------|---------:|:---------:|",
    ]
    for name in DATASETS:
        for key in ("sbp", "dbp", "map"):
            pressure = results["datasets"][name]["pressure"][key]
            signed = pressure["signed"]
            lines.append(
                f"| {LABELS[name]} | {PRESSURE_LABELS[key]} | {f3(signed['mean'])} ± "
                f"{f3(signed['sd'])} | **{pressure['aami']}** |"
            )

    lines += [
        "",
        "## Диапазоны и выбросы",
        "",
        "Выбросы считаются на уровне сегмента: waveform MAE выше 50/100/500",
        "мм рт. ст. и наличие prediction вне физиологического контрольного",
        "интервала [20, 300] мм рт. ст. Значения не удалялись из метрик.",
        "",
        "| Набор | GT min…max | Prediction min…max | MAE >50 | >100 | >500 | Pred вне [20,300] |",
        "|-------|-----------:|-------------------:|--------:|-----:|-----:|-------------------:|",
    ]
    for name in DATASETS:
        data = results["datasets"][name]
        ranges = data["range"]
        outliers = data["outliers"]
        lines.append(
            f"| {LABELS[name]} | {f3(ranges['gt_min'])}…{f3(ranges['gt_max'])} | "
            f"{f3(ranges['pred_min'])}…{f3(ranges['pred_max'])} | "
            f"{outliers['mae_gt50']} | {outliers['mae_gt100']} | {outliers['mae_gt500']} | "
            f"{outliers['pred_outside_20_300']} |"
        )

    lines += [
        "",
        "## Относительная деградация waveform MAE",
        "",
        f"- ABP_PPG / MIMIC corrected: **{f15(results['ratios']['abp_ppg_over_mimic'])}×**.",
        f"- VitalDB / MIMIC corrected: **{f15(results['ratios']['vitaldb_over_mimic'])}×**.",
        "",
        "## Отдельное воспроизведение ошибочного расчёта публикации",
        "",
        "Таблица ниже не входит в основное сравнение. Она применяет к",
        "`test_original.p` формулу ×200+50 для GT и prediction, повторяя",
        "ошибку `evaluate.py`. Deviation = reproduced − paper.",
        "",
        "| Метрика | Paper | Reproduced ×200 | Deviation | |Deviation| |",
        "|---------|------:|----------------:|----------:|------------:|",
    ]
    for key, label in (("waveform", "Waveform MAE"), ("sbp", "SBP MAE"), ("dbp", "DBP MAE"), ("map", "MAP MAE")):
        row = results["publication_reproduction_x200"][key]
        lines.append(
            f"| {label} | {f3(row['paper'])} | {f6(row['reproduced'])} | "
            f"{f6(row['deviation'])} | {f6(row['absolute_deviation'])} |"
        )

    lines += [
        "",
        "## Проверки воспроизводимости",
        "",
        f"- Corrected MIMIC: `{tuple(verification['shape'])}`, `{verification['dtype']}`; порядок HDF5 сохранён.",
        f"- Максимальная ошибка обратного преобразования corrected GT: `{verification['gt_roundtrip_max_abs']:.3e}` мм рт. ст.",
        f"- Максимальная ошибка обратного преобразования corrected PPG: `{verification['ppg_roundtrip_max_abs']:.3e}`.",
        f"- Исходный GT точно соответствует делителю `{verification['original_gt_scale']:.13f}`; max error `{verification['original_gt_roundtrip_max_abs']:.3e}`.",
        f"- Corrected и original PPG идентичны после float32: `{verification['original_ppg_float32_equal']}`.",
        "- Метрики повторно вычисляются из локальных `pred_norm`-кэшей; внутренняя",
        "  проверка подтверждает неизменность данных, весов и `meta9.p`.",
        "- Corrected dataset и кэши находятся под игнорируемым `codes/data/`;",
        "  `test_original.p` и веса не изменялись.",
        "",
        "## Ограничения",
        "",
        "Доверительные интервалы, case-level статистика и независимая проверка",
        "нормализации VitalDB не входят в этот расчёт. DOCX и Markdown статьи не",
        "изменялись.",
        "",
    ]
    return "\n".join(lines)


def save_results(results: dict[str, Any], verification: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT_PATH.write_text(render_report(results, verification), encoding="utf-8")


def assert_nested_close(actual: Any, expected: Any, path: str = "results") -> None:
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: keys differ")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-10):
            raise AssertionError(f"{path}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def verify_saved_results(batch_size: int = 256) -> dict[str, Any]:
    """Recompute from pred_norm caches and require exact agreement with outputs."""
    verification = verify_corrected()
    saved = json.loads((CACHE_DIR / "metrics.json").read_text(encoding="utf-8"))
    recomputed, statuses = evaluate_all(batch_size=batch_size, force_inference=False)
    assert_nested_close(recomputed, saved)
    expected_report = render_report(recomputed, verification)
    actual_report = REPORT_PATH.read_text(encoding="utf-8")
    if actual_report != expected_report:
        raise AssertionError("Tracked Markdown report differs from recomputed results")
    if set(statuses.values()) != {"cache"}:
        raise AssertionError(f"Verification unexpectedly ran inference: {statuses}")
    return {
        "cache_status": statuses,
        "metrics_match": True,
        "report_match": True,
        **verification,
    }


def run(
    batch_size: int = 256,
    force_data: bool = False,
    force_inference: bool = False,
    write_report: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    verification = prepare_corrected(force=force_data)
    results, statuses = evaluate_all(
        batch_size=batch_size,
        force_inference=force_inference,
    )
    if write_report:
        save_results(results, verification)
    return results, verification, statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"), nargs="?", default="run")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    if args.command == "prepare":
        print(json.dumps(prepare_corrected(force=args.force_data), indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_saved_results(batch_size=args.batch_size), indent=2))
    else:
        results, verification, statuses = run(
            batch_size=args.batch_size,
            force_data=args.force_data,
            force_inference=args.force_inference,
            write_report=not args.no_report,
        )
        print(json.dumps({
            "waveform_mae": {
                name: results["datasets"][name]["waveform"] for name in DATASETS
            },
            "ratios": results["ratios"],
            "cache_status": statuses,
            "verification": verification,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
