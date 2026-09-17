"""
Full MIMIC test.p evaluation with the same metrics pipeline.
test.p contains normalized data: X_test, Y_test in [0, 1].
"""
import numpy as np
import pickle
import os
import torch
from _models_pytorch import UNetDS64, MultiResUNet1D

SEGMENT_LENGTH = 1024
BATCH_SIZE = 500

# Load
meta = pickle.load(open(os.path.join("data", "meta9.p"), "rb"))
min_ppg, max_ppg = meta["min_ppg"], meta["max_ppg"]
min_abp, max_abp = meta["min_abp"], meta["max_abp"]

dt = pickle.load(open(os.path.join("data", "test.p"), "rb"))
X_test = dt["X_test"]  # (N, 1024, 1) normalized
Y_test = dt["Y_test"]  # (N, 1024, 1) normalized
N = len(X_test)
print(f"MIMIC test.p: {N} segments")
print(f"X_test: {X_test.shape}, range=[{X_test.min():.4f}, {X_test.max():.4f}]")
print(f"Y_test: {Y_test.shape}, range=[{Y_test.min():.4f}, {Y_test.max():.4f}]")

# Denormalize ground truth to mmHg
Y_mmhg = Y_test[:, :, 0] * (max_abp - min_abp) + min_abp  # (N, 1024)

# Inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

approx_model = UNetDS64(in_channels=1, out_channels=1).to(device)
approx_model.load_state_dict(
    torch.load("models/approximate_state_dict.pth", map_location=device, weights_only=True))
approx_model.eval()

refine_model = MultiResUNet1D(n_channel=1).to(device)
refine_model.load_state_dict(
    torch.load("models/refinement_state_dict.pth", map_location=device, weights_only=True))
refine_model.eval()

X_t = torch.tensor(X_test, dtype=torch.float32)
preds = []
print(f"Running inference on {N} segments...")
for start in range(0, N, BATCH_SIZE):
    batch = X_t[start:start + BATCH_SIZE].to(device)
    with torch.no_grad():
        approx_out = approx_model(batch)
        y = approx_out[0].permute(0, 2, 1).contiguous()
        refined = refine_model(y)
        refined = refined.permute(0, 2, 1).contiguous()
    pred_batch = refined.cpu().numpy() * (max_abp - min_abp) + min_abp
    preds.append(pred_batch[:, :, 0])

pred_mmhg = np.concatenate(preds, axis=0)
print(f"Predictions: shape={pred_mmhg.shape}, range=[{pred_mmhg.min():.1f}, {pred_mmhg.max():.1f}] mmHg")

# Metrics
def compute_metrics(pred, gt, label=""):
    N = len(pred)
    mae_seg = np.mean(np.abs(pred - gt), axis=1)
    sbp_err = np.abs(pred.max(axis=1) - gt.max(axis=1))
    dbp_err = np.abs(pred.min(axis=1) - gt.min(axis=1))
    map_err = np.abs(pred.mean(axis=1) - gt.mean(axis=1))

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

    print(f"\n{'=' * 60}")
    print(f"{label} (N={N})")
    print(f"{'=' * 60}")
    print(f"Waveform MAE: {mae_seg.mean():.2f} +/- {mae_seg.std():.2f} mmHg")
    print(f"  median: {np.median(mae_seg):.2f}, P95: {np.percentile(mae_seg, 95):.2f}")
    print(f"SBP err:  {sbp_err.mean():.2f} +/- {sbp_err.std():.2f} mmHg")
    print(f"DBP err:  {dbp_err.mean():.2f} +/- {dbp_err.std():.2f} mmHg")
    print(f"MAP err:  {map_err.mean():.2f} +/- {map_err.std():.2f} mmHg")

    print(f"\nBHS Grading:")
    print(f"{'':>4} {'<=5':>8} {'<=10':>8} {'<=15':>8} {'Grade':>8}")
    for name, b, g in [("SBP", sbp_bhs, bhs_grade(*sbp_bhs)),
                        ("DBP", dbp_bhs, bhs_grade(*dbp_bhs)),
                        ("MAP", map_bhs, bhs_grade(*map_bhs))]:
        print(f"{name:>4} {b[0]:>7.1f}% {b[1]:>7.1f}% {b[2]:>7.1f}% {g:>8}")

    sbp_signed = pred.max(axis=1) - gt.max(axis=1)
    dbp_signed = pred.min(axis=1) - gt.min(axis=1)
    map_signed = pred.mean(axis=1) - gt.mean(axis=1)
    print(f"\nAAMI (ME +/- SD):")
    print(f"  SBP: {sbp_signed.mean():.2f} +/- {sbp_signed.std():.2f} mmHg")
    print(f"  DBP: {dbp_signed.mean():.2f} +/- {dbp_signed.std():.2f} mmHg")
    print(f"  MAP: {map_signed.mean():.2f} +/- {map_signed.std():.2f} mmHg")

compute_metrics(pred_mmhg, Y_mmhg, label="MIMIC TEST (full)")
