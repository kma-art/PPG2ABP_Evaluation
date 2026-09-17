"""
Evaluate PyTorch model on the ORIGINAL test.p (27 260 segments).
Determines correct denormalization (x150 vs x200) and compares with paper metrics.

Run: conda activate marl && cd codes && python _eval_original.py
"""
import numpy as np
import pickle
import os
import torch
from _models_pytorch import UNetDS64, MultiResUNet1D

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Working directory: {os.getcwd()}")

SEGMENT_LENGTH = 1024
BATCH_SIZE = 500

# ── Step 1: Load data and check normalization ────────────────────────────────

meta = pickle.load(open(os.path.join("data", "meta9.p"), "rb"))
min_ppg, max_ppg = meta["min_ppg"], meta["max_ppg"]
min_abp, max_abp = meta["min_abp"], meta["max_abp"]
print(f"[meta9] min_abp={min_abp}, max_abp={max_abp}, min_ppg={min_ppg}, max_ppg={max_ppg}")
print(f"  max_abp - min_abp = {max_abp - min_abp}")

dt = pickle.load(open(os.path.join("data", "test_original.p"), "rb"))
X_test = dt["X_test"]  # (N, 1024, 1) normalized
Y_test = dt["Y_test"]  # (N, 1024, 1) normalized
N = len(X_test)
print(f"\nOriginal test.p: {N} segments")
print(f"X_test: shape={X_test.shape}, dtype={X_test.dtype}, range=[{X_test.min():.6f}, {X_test.max():.6f}]")
print(f"Y_test: shape={Y_test.shape}, dtype={Y_test.dtype}, range=[{Y_test.min():.6f}, {Y_test.max():.6f}]")

# ── Step 1b: Determine normalization ─────────────────────────────────────────
Y_flat = Y_test[:, :, 0]

# Variant A: normalized as (y - 50) / 150  →  denorm: y_norm * 150 + 50
abp_A = Y_flat * 150 + 50
# Variant B: normalized as (y - 50) / 200  →  denorm: y_norm * 200 + 50
abp_B = Y_flat * 200 + 50

print(f"\n{'='*60}")
print(f"  Normalization check")
print(f"{'='*60}")
print(f"Variant A (x150+50): min={abp_A.min():.1f}, max={abp_A.max():.1f}, mean={abp_A.mean():.1f}")
print(f"  Per-segment SBP: mean={abp_A.max(axis=1).mean():.1f}, min={abp_A.max(axis=1).min():.1f}, max={abp_A.max(axis=1).max():.1f}")
print(f"  Per-segment DBP: mean={abp_A.min(axis=1).mean():.1f}, min={abp_A.min(axis=1).min():.1f}, max={abp_A.min(axis=1).max():.1f}")

print(f"Variant B (x200+50): min={abp_B.min():.1f}, max={abp_B.max():.1f}, mean={abp_B.mean():.1f}")
print(f"  Per-segment SBP: mean={abp_B.max(axis=1).mean():.1f}, min={abp_B.max(axis=1).min():.1f}, max={abp_B.max(axis=1).max():.1f}")
print(f"  Per-segment DBP: mean={abp_B.min(axis=1).mean():.1f}, min={abp_B.min(axis=1).min():.1f}, max={abp_B.min(axis=1).max():.1f}")

# Physiological ABP: SBP typically 90-180, DBP typically 50-100, extreme up to ~250
# If variant A gives max ~200 and variant B gives max ~250, variant A is more likely
print(f"\nPhysiological check:")
print(f"  Variant A: {(abp_A.max(axis=1) <= 250).mean()*100:.1f}% segments with SBP<=250")
print(f"  Variant B: {(abp_B.max(axis=1) <= 250).mean()*100:.1f}% segments with SBP<=250")
print(f"  Variant A: {(abp_A.min(axis=1) >= 30).mean()*100:.1f}% segments with DBP>=30")
print(f"  Variant B: {(abp_B.min(axis=1) >= 30).mean()*100:.1f}% segments with DBP>=30")

# ── Step 2: PyTorch inference ────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  PyTorch inference on {N} segments (device: {device})")
print(f"{'='*60}")

approx_model = UNetDS64(in_channels=1, out_channels=1).to(device)
approx_model.load_state_dict(
    torch.load("models/approximate_state_dict.pth", map_location=device, weights_only=True))
approx_model.eval()

refine_model = MultiResUNet1D(n_channel=1).to(device)
refine_model.load_state_dict(
    torch.load("models/refinement_state_dict.pth", map_location=device, weights_only=True))
refine_model.eval()

X_t = torch.tensor(X_test, dtype=torch.float32)
preds_norm = []  # keep normalized predictions
print(f"Running inference...")
for i, start in enumerate(range(0, N, BATCH_SIZE)):
    batch = X_t[start:start + BATCH_SIZE].to(device)
    with torch.no_grad():
        approx_out = approx_model(batch)
        y = approx_out[0].permute(0, 2, 1).contiguous()
        refined = refine_model(y)
        refined = refined.permute(0, 2, 1).contiguous()
    preds_norm.append(refined.cpu().numpy()[:, :, 0])
    if (i + 1) % 10 == 0:
        done = min(start + BATCH_SIZE, N)
        print(f"  {done}/{N} segments ({done*100//N}%)")

pred_norm = np.concatenate(preds_norm, axis=0)  # (N, 1024) normalized
print(f"Predictions: shape={pred_norm.shape}, norm range=[{pred_norm.min():.4f}, {pred_norm.max():.4f}]")

# ── Step 3: Metrics with both denormalizations ───────────────────────────────

def compute_metrics(pred_mmhg, gt_mmhg, label):
    """Compute waveform MAE, SBP/DBP/MAP errors, BHS grades, AAMI."""
    mae_seg = np.mean(np.abs(pred_mmhg - gt_mmhg), axis=1)
    sbp_err = np.abs(pred_mmhg.max(axis=1) - gt_mmhg.max(axis=1))
    dbp_err = np.abs(pred_mmhg.min(axis=1) - gt_mmhg.min(axis=1))
    map_err = np.abs(pred_mmhg.mean(axis=1) - gt_mmhg.mean(axis=1))

    def bhs(err):
        return ((err <= 5).mean()*100, (err <= 10).mean()*100, (err <= 15).mean()*100)

    def bhs_grade(pct5, pct10, pct15):
        if pct5 >= 60 and pct10 >= 85 and pct15 >= 95: return "A"
        elif pct5 >= 50 and pct10 >= 75 and pct15 >= 90: return "B"
        elif pct5 >= 40 and pct10 >= 65 and pct15 >= 85: return "C"
        else: return "D"

    sbp_bhs = bhs(sbp_err)
    dbp_bhs = bhs(dbp_err)
    map_bhs = bhs(map_err)

    sbp_signed = pred_mmhg.max(axis=1) - gt_mmhg.max(axis=1)
    dbp_signed = pred_mmhg.min(axis=1) - gt_mmhg.min(axis=1)
    map_signed = pred_mmhg.mean(axis=1) - gt_mmhg.mean(axis=1)

    print(f"\n{'='*65}")
    print(f"  {label} (N={len(pred_mmhg)})")
    print(f"{'='*65}")
    print(f"Waveform MAE: {mae_seg.mean():.3f} +/- {mae_seg.std():.3f} mmHg")
    print(f"  median: {np.median(mae_seg):.3f}, P95: {np.percentile(mae_seg, 95):.3f}")
    print(f"Pred range: [{pred_mmhg.min():.1f}, {pred_mmhg.max():.1f}] mmHg")
    print(f"GT   range: [{gt_mmhg.min():.1f}, {gt_mmhg.max():.1f}] mmHg")

    print(f"\n{'':>6} {'MAE':>10} {'BHS':^35} {'AAMI (ME±SD)':^25}")
    print(f"{'':>6} {'mean±SD':>10} {'<=5':>8} {'<=10':>8} {'<=15':>8} {'Grade':>6} {'ME':>10} {'SD':>8}")
    for name, abs_e, signed_e, b, g in [
        ("SBP", sbp_err, sbp_signed, sbp_bhs, bhs_grade(*sbp_bhs)),
        ("DBP", dbp_err, dbp_signed, dbp_bhs, bhs_grade(*dbp_bhs)),
        ("MAP", map_err, map_signed, map_bhs, bhs_grade(*map_bhs)),
    ]:
        print(f"  {name:>4} {abs_e.mean():>5.3f}±{abs_e.std():<5.3f}"
              f" {b[0]:>7.1f}% {b[1]:>7.1f}% {b[2]:>7.1f}% {g:>6}"
              f" {signed_e.mean():>+10.3f} {signed_e.std():>8.3f}")

    return {
        "waveform_mae": mae_seg.mean(),
        "waveform_mae_std": mae_seg.std(),
        "sbp_mae": sbp_err.mean(), "sbp_std": sbp_err.std(),
        "dbp_mae": dbp_err.mean(), "dbp_std": dbp_err.std(),
        "map_mae": map_err.mean(), "map_std": map_err.std(),
        "sbp_bhs": sbp_bhs, "dbp_bhs": dbp_bhs, "map_bhs": map_bhs,
    }

# Variant A: denorm x150 + 50
gt_A = Y_flat * 150 + 50
pred_A = pred_norm * 150 + 50
res_A = compute_metrics(pred_A, gt_A, "Variant A: denorm x150+50")

# Variant B: denorm x200 + 50
gt_B = Y_flat * 200 + 50
pred_B = pred_norm * 200 + 50
res_B = compute_metrics(pred_B, gt_B, "Variant B: denorm x200+50")

# Also try what evaluate.py actually does:
# It computes error in normalized space then multiplies by max_abp (200)
# This is: |y_pred_norm - y_gt_norm| * 200
# Which equals denorm x200 error (since offset cancels)
# So Variant B IS what evaluate.py produces

# ── Step 4: Compare with paper ───────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  Paper comparison (27 260 segments)")
print(f"{'='*65}")
print(f"{'':>20} {'Paper':>12} {'Var A (x150)':>14} {'Var B (x200)':>14}")
print(f"  {'Waveform MAE':>18} {'4.604':>12} {res_A['waveform_mae']:>14.3f} {res_B['waveform_mae']:>14.3f}")
print(f"  {'DBP MAE':>18} {'3.449':>12} {res_A['dbp_mae']:>14.3f} {res_B['dbp_mae']:>14.3f}")
print(f"  {'DBP SD':>18} {'6.147':>12} {res_A['dbp_std']:>14.3f} {res_B['dbp_std']:>14.3f}")
print(f"  {'MAP MAE':>18} {'2.310':>12} {res_A['map_mae']:>14.3f} {res_B['map_mae']:>14.3f}")
print(f"  {'MAP SD':>18} {'4.437':>12} {res_A['map_std']:>14.3f} {res_B['map_std']:>14.3f}")
print(f"  {'SBP MAE':>18} {'5.727':>12} {res_A['sbp_mae']:>14.3f} {res_B['sbp_mae']:>14.3f}")
print(f"  {'SBP SD':>18} {'9.162':>12} {res_A['sbp_std']:>14.3f} {res_B['sbp_std']:>14.3f}")

# Determine which variant matches paper better
paper = {"waveform_mae": 4.604, "dbp_mae": 3.449, "map_mae": 2.310, "sbp_mae": 5.727}
err_A = sum(abs(res_A[k] - v) for k, v in paper.items())
err_B = sum(abs(res_B[k] - v) for k, v in paper.items())
print(f"\nTotal absolute deviation from paper:")
print(f"  Variant A (x150): {err_A:.3f}")
print(f"  Variant B (x200): {err_B:.3f}")
winner = "A (x150)" if err_A < err_B else "B (x200)"
print(f"  -> Best match: Variant {winner}")

if err_A < err_B:
    print(f"\n  CONCLUSION: Original test.p uses normalization (y-50)/150,")
    print(f"  i.e. (y - min_abp) / (max_abp - min_abp). This is CORRECT.")
    print(f"  evaluate.py's x200 denormalization is BUGGED (inflates errors by x1.333).")
else:
    print(f"\n  CONCLUSION: Original test.p uses normalization (y-50)/200,")
    print(f"  i.e. y / max_abp. evaluate.py's x200 denormalization is CORRECT.")
