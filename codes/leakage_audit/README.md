# Independence audit of the original PPG2ABP test set

These scripts check whether the MIMIC-III test set used in the original PPG2ABP
publication can be treated as independent of the data used for training. They
read only artifacts that the original authors distribute, plus — for two
optional checks — files regenerated locally with the authors' own
`data_handling.py`.

The conclusion they support is narrow and deliberately so: **independence
cannot be confirmed from the published material.** Nothing here asserts what
any particular training run consumed; the original authors publish no training
log.

## The checks

| Script | Reads | Answers |
|---|---|---|
| `check_g_fresh.py` | `data.hdf5` | How many rows are byte-identical repeats, and how many test rows have a copy among the rows the split assigns to training and validation. Also confirms the file hash and that the min/max of rows 0–99 999 reproduce `meta9.p`. |
| `check_h_allfolds.py` | `data.hdf5` | Whether that share depends on the cross-validation fold. It does not: the same 16 265 test rows are affected for every `fold_id`. |
| `check_f_candidates.py` | `candidates.p` | How often the same episode appears in the direct output of `downsample_data()`, and what overlap that predicts after a shuffle and an index cut. |
| `check_i_phases.py` | `candidates.p` | Whether the file has the exact prefix/suffix structure the two-pass binning in `downsample_data()` predicts. |
| `check_e_mapping.py` *(optional)* | `data.hdf5`, `train9.p`, `val9.p`, `meta9.p` | Whether a locally regenerated split is exactly the HDF5 rows the index arithmetic says it should be. |
| `check_d_pickles.py` *(optional)* | `train9.p`, `val9.p`, `test_original.p` | Whether the overlap survives in the arrays actually fed to the model, where the two parts carry different normalization. |

`check_d_pickles.py` compares segments through a fingerprint that is invariant
to per-channel scale and offset, because the training and test parts are
normalized with different min/max; a sample of the matches is then verified
strictly by fitting the affine relation, and random pairs serve as a negative
control.

## Inputs

Place the two published artifacts under `codes/data/`:

| File | Google Drive id | SHA-256 |
|---|---|---|
| `data.hdf5` | `1IxN2sX2TX0uK6CFDh8eudb8haz3RlF7X` | `4d1a3085cd91f8725682c9290c0527f28651f6ce290d6d2a2da796b316adbf25` |
| `candidates.p` | `1u-yvkqJmmrYCbuSw3lnS8mIIcEHRQxFD` | `4c22fe9d3ec1268cdaefe3220200e41b799e6702f8d231e51582ec82c3389a38` |

Both ids come from the original repository and notebook
(https://github.com/nibtehaz/PPG2ABP). `meta9.p` ships with this repository.
`train9.p`, `val9.p` and `test_original.p` are produced by running the authors'
`data_handling.py` over their `data.hdf5`; nobody distributes them.

Each script resolves every input in this order: an explicit command-line
option, then the environment variable listed in `_common.py`, then
`codes/data/<file name>`. So either of these works:

```bash
python codes/leakage_audit/check_g_fresh.py --hdf5 /elsewhere/data.hdf5
PPG2ABP_DATA_HDF5=/elsewhere/data.hdf5 python codes/leakage_audit/check_g_fresh.py
```

Run them from the repository root. `h5py` is required by the three scripts that
read HDF5.

## Expected output

Measured on the published `data.hdf5` and `candidates.p`:

- 127 260 rows, 78 898 distinct segments; 48 313 segments occur more than once,
  so 38 % of the rows are repeats;
- 16 265 of the 27 260 test rows (59.7 %) have a byte-identical copy among rows
  0–99 999, at 1024 samples and at the stored 1250 alike;
- that 59.7 % is identical for all ten folds; only its split between the
  training part (53.5–53.9 %) and the validation part (5.8–6.2 %) moves;
- `candidates.p` holds 127 260 entries and 78 926 distinct triples, the first
  78 926 pairwise distinct and the remaining 48 334 each repeating one of them,
  with no triple occurring more than twice;
- the overlap that structure predicts, 2 × 48 334 × p × (1 − p) with
  p = 27 260 / 127 260, is 16 271 — 0.04 % from the measured 16 265.

`check_g_fresh.py` hashes 2.5 GB and takes a few minutes; `--skip-hash` skips
that pass. Both HDF5 scripts hold one fingerprint per row in memory.
