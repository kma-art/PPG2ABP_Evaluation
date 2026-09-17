"""
Generate figures for the article.

Figure 1: Bar chart — MAE degradation across datasets.
Figure 2: Waveform examples — good (MIMIC), moderate (ABP_PPG), poor (VitalDB).

Run: conda activate marl && cd codes && python _article_figures.py

Output: codes/figures/figure_1.png, codes/figures/figure_2.png
"""

import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
from _models_pytorch import UNetDS64, MultiResUNet1D

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Journal formatting ────────────────────────────────────────────────────────
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Bar chart — MAE degradation
# ══════════════════════════════════════════════════════════════════════════════

def figure_1():
    datasets = ["MIMIC-III\n(in-domain)", "ABP_PPG\n(same-source)", "VitalDB\n(cross-domain)"]
    mae_mean = [4.61, 12.37, 25.92]
    mae_sd   = [5.05, 6.32, 16.60]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    bars = ax.bar(datasets, mae_mean, yerr=mae_sd, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.8,
                  error_kw={"linewidth": 1.2, "capthick": 1.2})

    # value labels on bars
    for bar, m, s in zip(bars, mae_mean, mae_sd):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 1.0,
                f"{m:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # multiplier annotations
    ax.annotate("", xy=(1, mae_mean[1] + mae_sd[1] + 4),
                xytext=(0, mae_mean[0] + mae_sd[0] + 4),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text(0.5, mae_mean[1] + mae_sd[1] + 5.5, r"$\times$2,7",
            ha="center", va="bottom", fontsize=10, color="gray")

    ax.annotate("", xy=(2, mae_mean[2] + mae_sd[2] + 4),
                xytext=(0, mae_mean[0] + mae_sd[0] + 4),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text(1.0, mae_mean[2] + mae_sd[2] + 5.5, r"$\times$5,6",
            ha="center", va="bottom", fontsize=10, color="gray")

    ax.set_ylabel("MAE, \u043c\u043c \u0440\u0442. \u0441\u0442.")
    ax.set_title("")
    ax.set_ylim(0, 55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(OUT_DIR, "figure_1.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] Figure 1 saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Waveform examples
# ══════════════════════════════════════════════════════════════════════════════

def load_models(device):
    approx = UNetDS64(in_channels=1, out_channels=1).to(device)
    approx.load_state_dict(
        torch.load("models/approximate_state_dict.pth", map_location=device, weights_only=True))
    approx.eval()

    refine = MultiResUNet1D(n_channel=1).to(device)
    refine.load_state_dict(
        torch.load("models/refinement_state_dict.pth", map_location=device, weights_only=True))
    refine.eval()
    return approx, refine


def predict_one(x_norm, approx, refine, device):
    """x_norm: (1024,) normalized PPG → pred_norm: (1024,)"""
    x_t = torch.tensor(x_norm.reshape(1, 1024, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        approx_out = approx(x_t)
        y = approx_out[0].permute(0, 2, 1).contiguous()
        refined = refine(y)
        refined = refined.permute(0, 2, 1).contiguous()
    return refined.cpu().numpy()[0, :, 0]


def find_median_segment(_, Y_mmhg, pred_mmhg):
    """Find segment closest to median MAE."""
    mae_per_seg = np.mean(np.abs(pred_mmhg - Y_mmhg), axis=1)
    median_mae = np.median(mae_per_seg)
    idx = np.argmin(np.abs(mae_per_seg - median_mae))
    return idx, mae_per_seg[idx]


CACHE_DIR = os.path.join(OUT_DIR, "_cache")


def run_inference_cached(name, data_file, denorm_fn, device, approx, refine):
    """Run inference with caching. Returns (gt_mmhg, pred_mmhg, mae_per_seg)."""
    cache_gt = os.path.join(CACHE_DIR, f"{name}_gt.npy")
    cache_pred = os.path.join(CACHE_DIR, f"{name}_pred.npy")

    if os.path.exists(cache_gt) and os.path.exists(cache_pred):
        print(f"  [{name}] Loading from cache...")
        return np.load(cache_gt), np.load(cache_pred)

    print(f"  [{name}] Running inference (will be cached)...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    dt = pickle.load(open(os.path.join("data", data_file), "rb"))
    X = dt["X_test"]
    Y = dt["Y_test"][:, :, 0]  # (N, 1024)

    preds = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for start in range(0, len(X), 500):
        batch = X_t[start:start+500].to(device)
        with torch.no_grad():
            ao = approx(batch)
            y = ao[0].permute(0, 2, 1).contiguous()
            r = refine(y)
            r = r.permute(0, 2, 1).contiguous()
        preds.append(r.cpu().numpy()[:, :, 0])
    pred_norm = np.concatenate(preds, axis=0)

    gt_mmhg, pred_mmhg = denorm_fn(Y, pred_norm)

    np.save(cache_gt, gt_mmhg)
    np.save(cache_pred, pred_mmhg)
    print(f"  [{name}] Cached to {CACHE_DIR}/")
    return gt_mmhg, pred_mmhg


def figure_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Figure 2] Device: {device}")

    meta = pickle.load(open(os.path.join("data", "meta9.p"), "rb"))
    min_abp, max_abp = meta["min_abp"], meta["max_abp"]
    abp_range = max_abp - min_abp  # ~150

    # Check if ALL caches exist — if so, skip model loading entirely
    all_cached = all(
        os.path.exists(os.path.join(CACHE_DIR, f"{n}_{s}.npy"))
        for n in ("mimic", "abp_ppg", "vitaldb") for s in ("gt", "pred")
    )

    if all_cached:
        print("  All predictions cached, skipping model load.")
        approx = refine = None
    else:
        approx, refine = load_models(device)

    t = np.arange(1024) / 125.0  # seconds

    # ── Run / load predictions ───────────────────────────────────────────────
    gt_m, pred_m = run_inference_cached(
        "mimic", "test_original.p",
        lambda Y, P: (Y * 200 + 50, P * 200 + 50),  # x200 canonical
        device, approx, refine)

    gt_a, pred_a = run_inference_cached(
        "abp_ppg", "abp_ppg_test.p",
        lambda Y, P: (Y, P * abp_range + min_abp),  # GT raw mmHg, pred x150
        device, approx, refine)

    gt_v, pred_v = run_inference_cached(
        "vitaldb", "vitaldb_test.p",
        lambda Y, P: (Y, P * abp_range + min_abp),  # GT raw mmHg, pred x150
        device, approx, refine)

    # ── Pick median segments ─────────────────────────────────────────────────
    panels = []
    for gt, pred, title in [
        (gt_m, pred_m, "\u0430) MIMIC-III (in-domain)"),
        (gt_a, pred_a, "\u0431) ABP_PPG (same-source)"),
        (gt_v, pred_v, "\u0432) VitalDB (cross-domain)"),
    ]:
        idx, mae = find_median_segment(None, gt, pred)
        label = f"MAE = {mae:.1f} \u043c\u043c \u0440\u0442. \u0441\u0442."
        panels.append((gt[idx], pred[idx], mae, title, label))

    # ── Draw ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for ax, (gt, pred, mae, title, label) in zip(axes, panels):
        ax.plot(t, gt, color="#2E86AB", linewidth=1.2, label="Ground truth")
        ax.plot(t, pred, color="#E8553A", linewidth=1.0, linestyle="--", label="\u041f\u0440\u0435\u0434\u0441\u043a\u0430\u0437\u0430\u043d\u0438\u0435")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("\u0410\u0414, \u043c\u043c \u0440\u0442. \u0441\u0442.")
        ax.text(0.97, 0.92, label, transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"))
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("\u0412\u0440\u0435\u043c\u044f, \u0441")
    fig.tight_layout(h_pad=1.5)

    path = os.path.join(OUT_DIR, "figure_2.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] Figure 2 saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    figure_1()
    figure_2()
    print("\nDone. Figures in codes/figures/")
