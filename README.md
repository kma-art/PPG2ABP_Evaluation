# Evaluation of the generalization ability of PPG2ABP

Code, canonical result tables and reproduction instructions for the article

> **«Оценка обобщающей способности модели PPG2ABP для прогнозирования
> артериального давления по фотоплетизмограмме»**
> (Evaluation of the generalization ability of the PPG2ABP model for arterial
> blood pressure prediction from photoplethysmography)

submitted to *Vrach i Informatsionnye Tekhnologii* («Врач и информационные
технологии»). **This release accompanies the revised manuscript** and replaces
the first-submission material published here earlier.

The repository contains a **PyTorch reimplementation** of the PPG2ABP
architecture (Ibtehaz N, Mahmud S, Chowdhury MEH, et al. Bioengineering.
2022;9(11):692 — https://github.com/nibtehaz/PPG2ABP) together with a
multi-dataset evaluation
pipeline (MIMIC-III, ABP_PPG, VitalDB) that measures how far the model's
accuracy carries over when the preprocessing and the source of the data differ
from the training distribution.

---

## What changed since the previous release

The earlier version of this repository described the first submission. Several
of its statements no longer hold, and are corrected here.

| Previously published | This release |
|---|---|
| The headline MIMIC-III figure used the `×200 + 50` denormalization. | All main results use `×149.98749589709124 + 50`, the inverse of the training normalization. `×200 + 50` is kept only to reproduce the original publication (§3.1 of the article), because the original evaluation script multiplies by `max_abp` instead of `max_abp − min_abp`. |
| The original MIMIC-III test set was treated as an independent test set. | Independence **cannot be confirmed** from the published material: 59.7 % of its segments have a byte-identical copy among the rows the authors' own split script assigns to training and validation. See [Independence of the original MIMIC-III test set](#independence-of-the-original-mimic-iii-test-set). |
| The pretrained weights were labelled "fold 9". | The original authors do not state which cross-validation fold produced the released weights. It changes no number here: the normalization parameters are identical for all ten folds. |
| VitalDB was represented by the original convenience sample of 9 400 segments. | A prespecified, subject-disjoint holdout of 9 600 segments from 96 further subjects was acquired after the normalization strategy and the analysis rules had been frozen; it is the confirmatory analysis. The original 9 400 segments remain as the exploratory sample. |
| Point estimates were segment-weighted, without interval estimates. | The primary estimand is cluster-equal (mean within a record or subject, then equal weight per cluster) with 95 % cluster-bootstrap confidence intervals over 10 000 replicates, Cliff's delta and Holm-adjusted Mann-Whitney U tests. Segment-weighted values are still reported alongside. |
| AAMI results were reported as "PASS" and "FAIL". | Results are reported as a comparison against the numeric thresholds. Neither the BHS grades nor the AAMI thresholds constitute a clinical or regulatory validation of a measuring device. |
| No machine-readable results were published. | The canonical bundle `revision_v2/point6/article_bundle_v2.json`, eleven derived tables and a standalone verifier are included. |

---

## Contents

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── article_revision_extended_materials.md — public S1-S9 reproducibility companion
├── verify_public_bundle.py         — standalone check of the published bundle and tables
│
├── codes/
│   ├── _models_pytorch.py          — UNetDS64 + MultiResUNet1D (PyTorch port)
│   ├── metrics.py                  — BHS helper
│   │
│   ├── _mimic_eval.py              — smoke test on the bundled 500-segment subset
│   ├── _eval_original.py           — §3.1 reproduction on the original MIMIC-III test set
│   ├── _unified_evaluation.py      — physical-scale evaluation across the three datasets
│   ├── segment_provenance.py       — maps every segment back to its source record or case
│   │
│   ├── _abp_ppg_pipeline.py        — ABP_PPG: build test set, infer, report metrics
│   ├── _abp_ppg_inference.ipynb    — the same pipeline as a notebook, with plots
│   │
│   ├── _vital_download.ipynb       — VitalDB: acquisition through the open API
│   ├── _vital_adapt.ipynb          — VitalDB: normalization ablation and diagnostics
│   ├── _vitaldb_bhs.py             — VitalDB: BHS and AAMI on the original 9 400 segments
│   │
│   ├── _article_figures.py         — regenerates the pre-revision figures 1 and 2
│   ├── test_unified_evaluation.py  — unit tests
│   ├── test_segment_provenance.py  — unit tests
│   │
│   ├── _revision_point4.ipynb      — thin notebook over the point4 results
│   ├── _revision_point5.ipynb      — thin notebook over the point5 results
│   ├── _article_revision_v2.ipynb  — thin notebook over the canonical bundle
│   │
│   ├── leakage_audit/              — independence audit of the original MIMIC-III test set
│   ├── tables/revision_v2/point6/  — the eleven published result tables (CSV and Markdown)
│   ├── figures/revision_v2/point5/ — diagnostic figures (errors, calibration, Bland-Altman)
│   └── data/
│       ├── meta9.p                 — training-set normalization parameters
│       └── test.p                  — 500-segment MIMIC-III subset, for the smoke test
│
└── revision_v2/                    — the reproducible analysis contour
    ├── point4/                     — VitalDB holdout: protocol, acquisition, inference, statistics
    ├── point5/                     — post-processing, correlation, calibration, agreement, figures
    └── point6/                     — canonical bundle and derived tables
```

Everything that ships in this repository is small. Model weights and the large
datasets are downloaded separately — see [Setup](#setup) and
[Downloads](#downloads).

Two notes on the shipped scripts. `codes/_unified_evaluation.py` and
`codes/segment_provenance.py` write their reports in Russian, the language of
the article, into a `.context/` directory they create; the reports are working
notes, and every number they contain also appears in the published tables. Both
mention "fold 9": that names the `meta9.p` normalization file, and is not an
attribution of the released weights to a particular cross-validation fold. All
files under `revision_v2/` and the evaluation scripts are shipped exactly as
they were run, without edits.

---

## Environment

```bash
# Python 3.12
pip install -r requirements.txt
```

The pinned versions are the ones used to produce the reported results. Two
entry points — `codes/_unified_evaluation.py` and everything under
`codes/leakage_audit/` — read HDF5 directly and need `h5py`, which was
installed in a separate environment; `requirements.txt` explains the split.

### Working directory

Two conventions coexist, and both matter:

| Command shape | Run it from |
|---|---|
| `python codes/<script>.py`, `python -m revision_v2.<point> ...`, `python verify_public_bundle.py` | the repository root |
| `python _eval_original.py`, `_mimic_eval.py`, `_abp_ppg_pipeline.py`, `_vitaldb_bhs.py`, `_article_figures.py`, and the notebooks | inside `codes/` |

The scripts in the second row resolve `data/...`, `models/...` and
`raw_data/...` relative to the current directory, so they expect `codes/` to be
the working directory. Everything else resolves paths from the repository root.

---

## Setup

### Model weights

The pretrained weights (about 86 MB for the PyTorch pair) are not stored in the
repository. Download them from the project folder on Google Drive and place
them under `codes/models/`:

> **Project Google Drive folder:**
> https://drive.google.com/drive/folders/1duatl87TosXDbHLa8gb6fHu7VoBLhMrb

```
codes/models/approximate_state_dict.pth   — Approximate Network (PyTorch port)
codes/models/refinement_state_dict.pth    — Refinement Network (PyTorch port)
```

The same folder carries the original Keras weights (`ApproximateNetwork.h5`,
`RefinementNetwork.h5`) that the port was converted from, and the large
prepared datasets listed under [Downloads](#downloads).

### Smoke test

With the two `.pth` files in place:

```bash
cd codes
python _mimic_eval.py
```

Verified output on the bundled 500-segment subset: waveform MAE
6.02 ± 4.79 mmHg, predictions within [50.5, 193.7] mmHg. This is **not** a
result reported in the article — it only confirms that the weights load and
that the port produces predictions on the expected scale. The article's
MIMIC-III figures come from the full 27 260-segment test set below.

---

## Reproducing the results

The pipeline is layered. Each layer states the artifact it needs and where that
artifact comes from.

### Layer 1 — §3.1: the original MIMIC-III test set (27 260 segments)

Required: `codes/data/test_original.p` (about 446 MB), the canonical
Ibtehaz and Rahman test split with normalized PPG and ABP. Download it from the
project Drive folder into `codes/data/`, then:

```bash
cd codes
python _eval_original.py
```

The script runs inference over the full test set and prints the comparison with
the original publication next to a rounded `×150 + 50` column.

The values the article reports for this reproduction come from the canonical
contour (`legacy_audits.csv`, `audit_type = publication_x200_reproduction`):
with `×200 + 50`, waveform 4.605, SBP 5.731, DBP 3.451 and MAP 2.309 mmHg
against the published 4.604, 5.727, 3.449 and 2.310, the largest deviation
being 0.004 mmHg. The article's main MIMIC-III numbers are not on that scale:
they use the exact inverse of the training normalization,
`×149.98749589709124 + 50`, and are produced by `codes/_unified_evaluation.py`
together with the `revision_v2` contour in Layer 4, which is also where the
cluster-equal estimand and the confidence intervals come from.

**Regenerating it instead of downloading.** Take the four MIMIC-III subset
files `Part_1.mat` ... `Part_4.mat` (3.4 GB) from the UCI Machine Learning
Repository (Kachuee distribution,
https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation),
place them under `codes/raw_data/`, and run the **original authors'**
preprocessing from https://github.com/nibtehaz/PPG2ABP/tree/master/codes:
`data_processing.py` produces `data/data.hdf5`, then `data_handling.py`
produces the ten-fold split and `data/test.p`; rename the latter to
`codes/data/test_original.p`. Those scripts target TensorFlow 1.5 and
Keras 2.2 and need their own Python environment; they are used only for data
preparation, and nothing is trained here.

Note that `data_processing.py` fixes no random seed, so a regenerated
`data.hdf5` will not be identical to the one the original authors distribute.
For the audit below, use their published file.

### Layer 2 — ABP_PPG (11 706 segments, same source, different preprocessing)

An alternative MIMIC-III preprocessing that uses pulse-wave delineation instead
of the Kachuee segmentation.

1. Download the ABP_PPG raw `.mat` archive (5 856 files) from Zenodo,
   https://zenodo.org/records/4598938, and extract it to
   `codes/raw_data/ABP_PPG/`.
2. Build the test set, run inference and print the metrics:

   ```bash
   cd codes
   python _abp_ppg_pipeline.py
   ```

   This writes `codes/data/abp_ppg_test.p`. The notebook
   `_abp_ppg_inference.ipynb` runs the same pipeline with diagnostic plots.

### Layer 3 — VitalDB (cross-domain)

Fully external data: Seoul National University Bundang Hospital, surgical
patients, SNUADC monitor.

*Original exploratory sample, 9 400 segments from 94 subjects* — reproducible
from the open VitalDB API, with no download from this project required. As a
shortcut, the prepared `vitaldb_raw.p` (about 154 MB) is in the project Drive
folder; place it at `codes/raw_data/vitaldb_raw.p` and skip the acquisition
notebook.

```bash
cd codes
jupyter nbconvert --execute --to notebook --inplace _vital_download.ipynb
python _vitaldb_bhs.py
```

`_vital_adapt.ipynb` compares the three PPG normalization strategies (min-max
over VitalDB's own range; the MIMIC training parameters applied directly;
alignment on the 1st and 99th percentiles) and writes
`codes/data/vitaldb_test.p` for figure generation. The strategy was chosen on
this sample, so its results are exploratory.

*Independent holdout, 9 600 segments from 96 further subjects* — the
confirmatory analysis. The selection protocol, the normalization parameters and
the processing rules were frozen **before** any holdout signal was downloaded;
the frozen protocol is published as `revision_v2/point4/protocol_v1.json`. From
the repository root:

```bash
python -m revision_v2.point4 protocol      # re-derive and check the frozen protocol
python -m revision_v2.point4 acquire       # download through the VitalDB API
python -m revision_v2.point4 infer         # PyTorch inference, no re-selection
python -m revision_v2.point4 statistics    # cluster-equal estimates and bootstrap CIs
python -m revision_v2.point4 report
python -m revision_v2.point4 verify
```

The subjects of the original sample are excluded by construction, so the two
VitalDB samples are subject-disjoint.

### Layer 4 — statistics, diagnostics and the published tables

Two preparatory steps feed the aggregation. `_unified_evaluation.py` evaluates
all three datasets in the physical scale and caches the predictions;
`segment_provenance.py` maps every segment back to the record or case it came
from, which is what the cluster-equal estimand needs. Both write their working
reports in Russian under `.context/`.

```bash
python codes/_unified_evaluation.py run    # needs h5py; writes the prediction caches
python codes/segment_provenance.py all     # segment-to-record manifests, private
```

```bash
python -m revision_v2.point5 analyze   # post-processing, correlation, calibration, agreement, robust tails
python -m revision_v2.point5 figures   # diagnostic figures
python -m revision_v2.point5 verify

python -m revision_v2.point6 build     # canonical bundle and the eleven derived tables
python -m revision_v2.point6 verify
```

`point6` performs no inference and no acquisition: it aggregates the frozen
outputs of the earlier stages deterministically and writes
`revision_v2/point6/article_bundle_v2.json` together with
`codes/tables/revision_v2/point6/*.csv` and `*.md`.

---

## Independence of the original MIMIC-III test set

The original MIMIC-III test set is used in the article as an in-domain
reproducibility check and as an optimistic bound on attainable accuracy, not as
an independent evaluation. The reason is checkable from the published files
alone, and `codes/leakage_audit/` contains the checks.

In `data.hdf5`, the file the original authors distribute for running their
tests, 78 898 distinct segments occupy 127 260 rows: 48 313 of them appear more
than once, so 38 % of the rows are repeats. Their own `data_handling.py` splits
that file purely by row index — rows 0 to 99 999 for training and validation,
rows 100 000 to 127 259 for the test part — and copies of the same episode
therefore fall on both sides of the cut. The measured consequence is that
**16 265 of the 27 260 test segments (59.7 %) have a byte-identical copy among
rows 0 to 99 999**. The share is the same for all ten folds, because every
`fold_id` consumes rows 0 to 99 999 in full and only moves the boundary between
training and validation.

The repeats originate in `downsample_data()` in the authors'
`data_processing.py`, whose second binning pass appends an episode when a
look-up succeeds, that is when the episode is already in the candidate list.
The authors' own `candidates.p` shows exactly the structure this predicts: the
first 78 926 triples are pairwise distinct, the remaining 48 334 each repeat one
of them, and no triple occurs more than twice.

What this does and does not establish: it is a statement about the published
artifacts and about the behaviour of the published split script. The original
authors do not publish a training log, so nothing here asserts what any
particular training run consumed. The article states the finding in exactly
that form.

### Running the audit

Two files come from the original authors' Google Drive, both linked from their
repository and notebook; place them under `codes/data/`.

| File | Google Drive id | SHA-256 |
|---|---|---|
| `data.hdf5` | `1IxN2sX2TX0uK6CFDh8eudb8haz3RlF7X` | `4d1a3085cd91f8725682c9290c0527f28651f6ce290d6d2a2da796b316adbf25` |
| `candidates.p` | `1u-yvkqJmmrYCbuSw3lnS8mIIcEHRQxFD` | `4c22fe9d3ec1268cdaefe3220200e41b799e6702f8d231e51582ec82c3389a38` |

```bash
python codes/leakage_audit/check_g_fresh.py       # duplicates and overlap in data.hdf5
python codes/leakage_audit/check_h_allfolds.py    # the same overlap for each of the ten folds
python codes/leakage_audit/check_f_candidates.py  # repeated entries in candidates.p
python codes/leakage_audit/check_i_phases.py      # two-pass structure of candidates.p
```

Two further checks work on files regenerated locally with the authors'
`data_handling.py` (`train9.p`, `val9.p`, `test_original.p`); nobody
distributes those, and the checks are optional:

```bash
python codes/leakage_audit/check_e_mapping.py     # the regenerated split is exactly HDF5 rows
python codes/leakage_audit/check_d_pickles.py     # the overlap in the arrays fed to the model
```

Every script takes explicit path options and also honours the environment
variables listed in `codes/leakage_audit/_common.py`, so the inputs may live
anywhere. `check_g_fresh.py` hashes 2.5 GB and takes a few minutes; pass
`--skip-hash` to skip that pass.

---

## Canonical results

`revision_v2/point6/article_bundle_v2.json` is the machine-readable source for
every number in the article. The tables under
`codes/tables/revision_v2/point6/` are derived from it exactly. To check
that, with nothing but this repository:

```bash
python verify_public_bundle.py
```

It confirms that the bundle is in its canonical serialization, that all eleven
tables are exactly its derived form in both CSV and Markdown, that the declared
analysis invariants (seed 20260908, 10 000 bootstrap replicates, cluster-equal
primary estimand, `raw` primary scenario) are the ones the article reports, and
that no case, subject, record or cluster identifier is exposed.

| Table | Rows | What it holds |
|---|---:|---|
| `dataset_composition` | 6 | entities, clusters and evaluated segments per dataset |
| `main_metrics` | 48 | waveform, SBP, DBP and MAP error, segment-weighted and cluster-equal, with 95 % CI |
| `signed_errors` | 36 | ME ± SD for the AAMI comparison, with 95 % CI |
| `bhs_proportions` | 108 | error proportions at ≤ 5, ≤ 10 and ≤ 15 mmHg |
| `normalization_holdout` | 20 | holdout variants A, B and C and the prespecified comparisons (Cliff's delta, Holm-adjusted p) |
| `postprocessing` | 30 | raw, clipping, rejection, and the two truth-dependent diagnostics |
| `robust_tails` | 24 | quantiles and tail counts of the error distributions |
| `correlation_agreement` | 8 | Pearson, Spearman, R², calibration slope and intercept, Bland-Altman limits |
| `calibration_bins` | 112 | fixed-bin calibration on cluster means |
| `figure_index` | 8 | the diagnostic figures with size and DPI |
| `legacy_audits` | 7 | the `×200` reproduction of the original publication, and outlier-subset audits |

Which article table maps to which file:

| In the article | File |
|---|---|
| Table 2, main error metrics | `main_metrics.csv`, rows with `estimand = cluster_equal` |
| Table 3, BHS thresholds | `bhs_proportions.csv`, rows with `estimand = cluster_equal` |
| Extended materials S3 and S5, AAMI comparison | `signed_errors.csv`, rows with `estimand = cluster_equal` |
| §3.1, reproduction of the original publication | `legacy_audits.csv`, `audit_type = publication_x200_reproduction` |
| §3.2, sensitivity to post-processing | `postprocessing.csv` |
| §3.3, correlation, calibration and agreement | `correlation_agreement.csv`, `calibration_bins.csv` |

### What cannot be reproduced from this repository alone

`python -m revision_v2.point6 verify` rebuilds the bundle from the per-stage
aggregates and therefore needs the private artifacts under `codes/data/`:
per-segment prediction caches and the provenance manifests. Those are **not**
distributed, because they carry case, subject and record identifiers. Running
Layers 1 to 4 above regenerates them from the open sources.

`verify_public_bundle.py` is the part anyone can run immediately: it does not
rebuild the bundle, it checks that the published tables are exactly what the
published bundle says they are.

---

## Results summary

Waveform mean absolute error in the physically correct scale, cluster-equal
estimand, no post-processing, with 95 % cluster-bootstrap CI:

| Dataset | Clusters | Waveform MAE (mmHg) |
|---|---:|---|
| MIMIC-III, original test set | 7 353 records | 3.680 ± 3.486 [3.601; 3.761] |
| ABP_PPG, same source, different preprocessing | 1 716 subjects | 12.544 ± 5.127 [12.298; 12.794] |
| VitalDB, original exploratory sample | 94 subjects | 25.921 ± 4.965 [24.913; 26.939] |
| VitalDB, independent holdout | 96 subjects | 27.054 ± 8.199 [25.548; 28.871] |

Comparison against the numeric thresholds, same estimand: the share of segments
with an error ≤ 5 mmHg, and the resulting BHS grade.

| Metric | MIMIC-III | ABP_PPG | VitalDB holdout |
|---|---|---|---|
| SBP | 76.6 % / B | 16.7 % / D | 10.9 % / D |
| DBP | 87.2 % / A | 43.8 % / C | 27.1 % / D |
| MAP | 89.7 % / A | 34.4 % / D | 20.8 % / D |

The AAMI numeric thresholds are `|ME| ≤ 5` and `SD ≤ 8` mmHg. On MIMIC-III both
are met for all three scalar metrics. On ABP_PPG the SD threshold is exceeded
for DBP and MAP, and both thresholds are exceeded for SBP. On the VitalDB
holdout only MAP stays inside the ME threshold, and no metric meets both.

Two readings are required for these numbers. The MIMIC-III column is not an
independent evaluation — see the section above — so 3.680 mmHg is an optimistic
bound, and the ratios of 3.4× on ABP_PPG and 7.4× on the VitalDB holdout are
upper estimates of the relative degradation. And a comparison against the BHS
or AAMI numeric thresholds is not a clinical or regulatory validation of a
blood-pressure measuring device.

---

## Downloads

| Artifact | Where from |
|---|---|
| PyTorch weights `approximate_state_dict.pth`, `refinement_state_dict.pth` | project Drive folder — https://drive.google.com/drive/folders/1duatl87TosXDbHLa8gb6fHu7VoBLhMrb |
| Original Keras weights `ApproximateNetwork.h5`, `RefinementNetwork.h5` | project Drive folder, same link |
| MIMIC-III canonical test split `test_original.p` | project Drive folder, same link |
| VitalDB prepared segments `vitaldb_raw.p` | project Drive folder, same link; also reproducible through the open API |
| MIMIC-III, Kachuee subset, raw `.mat` | UCI ML Repository — https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation |
| ABP_PPG, delineation preprocessing | Zenodo — https://zenodo.org/records/4598938 |
| VitalDB | https://vitaldb.net through the `vitaldb` PyPI package |
| Original `data.hdf5` and `candidates.p`, for the audit | the original authors' Drive, ids in the audit section above |

The full MIMIC-III Waveform Database is available under PhysioNet credentialed
access at https://physionet.org/content/mimiciii/ but is not needed for any
experiment here.

---

## Extended materials

The public [S1-S9 reproducibility companion](article_revision_extended_materials.md)
contains the full composition tables, secondary normalization ablation,
bootstrap summaries, AAMI table, robust tails, diagnostic figures and the
no-inference reproduction sequence. It is a public reproducibility document,
not an official journal supplement. The stable tagged file is
https://github.com/kma-art/PPG2ABP_Evaluation/blob/revision-v2/article_revision_extended_materials.md;
the corresponding release page is
https://github.com/kma-art/PPG2ABP_Evaluation/releases/tag/revision-v2.

Every numerical value in S1-S9 is traceable to
`revision_v2/point6/article_bundle_v2.json` or to one of the eleven tables
derived from it. Run `python verify_public_bundle.py` from the repository root
to verify those public artifacts without model weights, private manifests or
new inference.

The two figures listed in `figure_index` as `legacy_review_required` come from
the pre-revision analysis and are deliberately not published in this release;
`codes/_article_figures.py` regenerates them for anyone who wants to compare.

---

## License and attribution

The PyTorch port of the architecture, the multi-dataset evaluation pipeline
(ABP_PPG ingestion, VitalDB acquisition and adaptation, the statistical
contour, the BHS and AAMI scoring), the independence audit, the
figure-generating code and the article itself are released under the **MIT
License** (see `LICENSE`).

The architecture, the training procedure that produced the pretrained weights,
the small `codes/data/test.p` reference set and the `data.hdf5` and
`candidates.p` artifacts examined in the audit originate from:

> Ibtehaz N, Mahmud S, Chowdhury MEH, Khandakar A, Salman Khan M, Ayari MA,
> Tahir AM, Rahman MS. PPG2ABP: Translating Photoplethysmogram (PPG) Signals to
> Arterial Blood Pressure (ABP) Waveforms. *Bioengineering*. 2022;9(11):692.
> doi:10.3390/bioengineering9110692 (preprint: *arXiv:2005.01669*, 2020).

Original repository, also MIT-licensed: https://github.com/nibtehaz/PPG2ABP

If you use this work, please cite both the original PPG2ABP paper and the
present article (citation to be added on publication).

## Contact

For questions about reproduction, please open an issue in this repository.
