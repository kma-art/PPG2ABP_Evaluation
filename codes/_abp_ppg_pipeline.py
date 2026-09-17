"""
ABP_PPG Dataset: Preprocessing + Inference + Metrics

MIMIC-III data preprocessed by a different pipeline (not Kachuee).
PPG scale [0, 4.002] matches MIMIC training data — no normalization adaptation needed.

Pipeline:
1. Load .mat files → extract PPG (row 0) and ABP (row 1)
2. Crop to 1024 samples
3. Filter artifacts (ABP min < 0, flat PPG)
4. Normalize PPG via meta9.p
5. Save as data/abp_ppg_test.p
6. Inference: ApproximateNetwork → RefinementNetwork
7. Metrics: Waveform MAE, SBP/DBP/MAP error, BHS/AAMI grading
"""
import numpy as np
import scipy.io
import pickle
import os
import glob
import torch
from _models_pytorch import UNetDS64, MultiResUNet1D

FS = 125
SEGMENT_LENGTH = 1024
BATCH_SIZE = 500

# ── 1. Load and Extract Segments ──
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

DATA_DIR = os.path.join("raw_data", "ABP_PPG")
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.mat")))
print(f"Total .mat files: {len(files)}")

meta = pickle.load(open(os.path.join("data", "meta9.p"), "rb"))
min_ppg, max_ppg = meta["min_ppg"], meta["max_ppg"]
min_abp, max_abp = meta["min_abp"], meta["max_abp"]
print(f"meta9.p: PPG=[{min_ppg:.4f}, {max_ppg:.4f}], ABP=[{min_abp:.2f}, {max_abp:.2f}]")

ppg_segments = []
abp_segments = []
errors = 0

for i, fpath in enumerate(files):
    if (i + 1) % 1000 == 0:
        print(f"  Loading: {i+1}/{len(files)}...")
    try:
        mat = scipy.io.loadmat(fpath)
        sp = mat["signal_processing"]
        for seg_idx in range(sp.shape[1]):
            seg = sp[0, seg_idx]
            sig = seg["signal"]
            if isinstance(sig, np.ndarray) and sig.dtype == object:
                sig = sig.flat[0]
            sig = np.asarray(sig, dtype=np.float64)

            if sig.ndim != 2 or sig.shape[0] < 2 or sig.shape[1] < SEGMENT_LENGTH:
                continue

            ppg_segments.append(sig[0, :SEGMENT_LENGTH])  # row 0 = PPG
            abp_segments.append(sig[1, :SEGMENT_LENGTH])  # row 1 = ABP
    except Exception:
        errors += 1

X_raw = np.array(ppg_segments, dtype=np.float64)
Y_raw = np.array(abp_segments, dtype=np.float64)

print(f"\nExtracted: {len(X_raw)} segments from {len(files) - errors} files ({errors} errors)")
print(f"X_raw (PPG): {X_raw.shape}, range=[{X_raw.min():.4f}, {X_raw.max():.4f}], mean={X_raw.mean():.4f}")
print(f"Y_raw (ABP): {Y_raw.shape}, range=[{Y_raw.min():.2f}, {Y_raw.max():.2f}], mean={Y_raw.mean():.2f}")

# ── 2. Filter Artifacts ──
print(f"\n{'=' * 70}")
print("2. FILTERING")
print("=" * 70)

ppg_std = X_raw.std(axis=1)
abp_min = Y_raw.min(axis=1)
abp_max = Y_raw.max(axis=1)

mask_ppg_flat = ppg_std >= 0.01
mask_abp_neg = abp_min >= 0
mask_abp_range = (abp_min >= 20) & (abp_max <= 300)
mask_nan = ~(np.isnan(X_raw).any(axis=1) | np.isnan(Y_raw).any(axis=1))
mask = mask_ppg_flat & mask_abp_neg & mask_abp_range & mask_nan

print(f"Total segments:         {len(X_raw)}")
print(f"  Flat PPG (std < 0.01):  {(~mask_ppg_flat).sum()} removed")
print(f"  ABP < 0 (artifact):     {(~mask_abp_neg).sum()} removed")
print(f"  ABP out of [20, 300]:   {(~mask_abp_range).sum()} removed")
print(f"  NaN:                    {(~mask_nan).sum()} removed")
print(f"  After filtering:        {mask.sum()} segments")

X_filt = X_raw[mask]
Y_filt = Y_raw[mask]

# ── 3. Normalize PPG ──
print(f"\n{'=' * 70}")
print("3. NORMALIZATION")
print("=" * 70)

X_norm = ((X_filt - min_ppg) / (max_ppg - min_ppg)).astype(np.float32)
frac_01 = ((X_norm >= 0) & (X_norm <= 1)).mean()
print(f"Normalized PPG: range=[{X_norm.min():.4f}, {X_norm.max():.4f}], mean={X_norm.mean():.4f}")
print(f"% in [0, 1]: {frac_01*100:.1f}%")
print(f"GT ABP (mmHg): range=[{Y_filt.min():.1f}, {Y_filt.max():.1f}], mean={Y_filt.mean():.1f}")

# ── 4. Save test data ──
print(f"\n{'=' * 70}")
print("4. SAVE TEST DATA")
print("=" * 70)

X_test = X_norm.reshape(-1, SEGMENT_LENGTH, 1)
Y_test = Y_filt.reshape(-1, SEGMENT_LENGTH, 1).astype(np.float32)

abp_ppg_data = {
    "X_test": X_test,
    "Y_test": Y_test,
    "source": "ABP_PPG (MIMIC-III, alternative preprocessing)",
    "fs": FS,
    "segment_length": SEGMENT_LENGTH,
    "n_segments": len(X_test),
    "normalization": "meta9p_direct",
    "normalization_params": {
        "method": "meta9p_direct",
        "min_ppg": float(min_ppg),
        "max_ppg": float(max_ppg),
    },
    "filtering": {
        "ppg_std_min": 0.01,
        "abp_min_threshold": 0,
        "abp_range": [20, 300],
    },
}

output_path = os.path.join("data", "abp_ppg_test.p")
with open(output_path, "wb") as f:
    pickle.dump(abp_ppg_data, f)
print(f"Saved to {output_path}: X_test={X_test.shape}, Y_test={Y_test.shape}")

# ── 5. Inference ──
print(f"\n{'=' * 70}")
print("5. INFERENCE")
print("=" * 70)

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
print("Models loaded.")

X_t = torch.tensor(X_norm.reshape(-1, SEGMENT_LENGTH, 1), dtype=torch.float32)
N = len(X_t)
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
    if (start // BATCH_SIZE + 1) % 5 == 0:
        print(f"  Batch {start // BATCH_SIZE + 1}/{(N + BATCH_SIZE - 1) // BATCH_SIZE}")

pred_mmhg = np.concatenate(preds, axis=0)
print(f"Predictions: shape={pred_mmhg.shape}, range=[{pred_mmhg.min():.1f}, {pred_mmhg.max():.1f}] mmHg")

# ── 6. Metrics ──
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
    print(f"  (AAMI requires |ME| <= 5 and SD <= 8)")

    return {"n": N, "mae": (mae_seg.mean(), mae_seg.std()),
            "mae_median": np.median(mae_seg),
            "sbp": (sbp_err.mean(), sbp_err.std()),
            "dbp": (dbp_err.mean(), dbp_err.std()),
            "map": (map_err.mean(), map_err.std()),
            "sbp_bhs": sbp_bhs, "dbp_bhs": dbp_bhs, "map_bhs": map_bhs,
            "sbp_grade": bhs_grade(*sbp_bhs),
            "dbp_grade": bhs_grade(*dbp_bhs),
            "map_grade": bhs_grade(*map_bhs),
            "mae_per_seg": mae_seg}

print(f"\n{'=' * 70}")
print("6. METRICS")
print("=" * 70)

# All segments
r_all = compute_metrics(pred_mmhg, Y_filt, label="ALL SEGMENTS")

# ABP in [50, 200]
mask_physio = (Y_filt.min(axis=1) >= 50) & (Y_filt.max(axis=1) <= 200)
r_physio = compute_metrics(
    pred_mmhg[mask_physio], Y_filt[mask_physio],
    label="ABP IN [50, 200] mmHg")

# Exclude top 5% MAE outliers
p95 = np.percentile(r_all["mae_per_seg"], 95)
mask_clean = r_all["mae_per_seg"] <= p95
r_clean = compute_metrics(
    pred_mmhg[mask_clean], Y_filt[mask_clean],
    label=f"EXCLUDING TOP 5% OUTLIERS (MAE <= {p95:.1f})")

# ABP in [50, 200] + exclude top 5%
mask_strict = mask_physio & mask_clean
r_strict = compute_metrics(
    pred_mmhg[mask_strict], Y_filt[mask_strict],
    label="ABP IN [50, 200] + EXCLUDE TOP 5%")

# ── 7. Correlation ──
print(f"\n{'=' * 70}")
print("7. CORRELATION: FEATURES vs MAE")
print("=" * 70)

mae_seg = r_all["mae_per_seg"]
for name, vals in [
    ("ppg_std", X_filt.std(axis=1)),
    ("ppg_range", X_filt.max(axis=1) - X_filt.min(axis=1)),
    ("abp_sbp", Y_filt.max(axis=1)),
    ("abp_dbp", Y_filt.min(axis=1)),
    ("abp_mean", Y_filt.mean(axis=1)),
    ("abp_range", Y_filt.max(axis=1) - Y_filt.min(axis=1)),
]:
    r = np.corrcoef(vals, mae_seg)[0, 1]
    print(f"  corr({name:>12}, MAE) = {r:.3f}")

# ── 8. Cross-dataset comparison ──
print(f"\n{'=' * 70}")
print("8. CROSS-DATASET COMPARISON")
print("=" * 70)

def fmt(ms):
    return f"{ms[0]:.2f} +/- {ms[1]:.2f}"

print(f"{'Dataset':>20} {'N':>8} {'MAE':>22} {'SBP':>6} {'DBP':>6} {'MAP':>6}")
print("-" * 70)
print(f"{'ABP_PPG (all)':>20} {r_all['n']:>8} {fmt(r_all['mae']):>22} "
      f"{r_all['sbp_grade']:>6} {r_all['dbp_grade']:>6} {r_all['map_grade']:>6}")
print(f"{'ABP_PPG [50,200]':>20} {r_physio['n']:>8} {fmt(r_physio['mae']):>22} "
      f"{r_physio['sbp_grade']:>6} {r_physio['dbp_grade']:>6} {r_physio['map_grade']:>6}")
print(f"{'ABP_PPG strict':>20} {r_strict['n']:>8} {fmt(r_strict['mae']):>22} "
      f"{r_strict['sbp_grade']:>6} {r_strict['dbp_grade']:>6} {r_strict['map_grade']:>6}")
print(f"\nReference:")
print(f"{'MIMIC (paper)':>20} {'27260':>8} {'7.84':>22} {'A':>6} {'A':>6} {'A':>6}")
print(f"{'VitalDB (A)':>20} {'9400':>8} {'25.92 +/- 16.60':>22} {'D':>6} {'D':>6} {'D':>6}")
