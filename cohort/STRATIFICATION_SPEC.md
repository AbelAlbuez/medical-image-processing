# BraTS-2024 GLI — Stratification Spec

Purpose: turn the full BraTS-2024 GLI training pool into a **measured, reproducible,
stratified cohort** for the classical-segmentation study and the Phase-2 prior methods.
The cohort is a *selection based on computed tumor properties*, not a hand-picked list of
IDs — every stratification variable is measured from the data by `stratify_brats.py`.

## Why stratify (and on what)

The pipeline's known behaviour is failure-mode-dependent, so the cohort must guarantee
coverage of the cases that stress each failure mode we characterized on the 20-case set:

| Stratification axis | Definition (measured) | Why it matters |
|---|---|---|
| **ET volume** | label-3 volume in mm³, binned absent / small (<p33) / medium / large (≥p66) using pool-wide percentiles | Small ET is the hardest for intensity methods; volume drives Dice sensitivity |
| **Focality** | # ET connected components (26-connectivity, ≥10 voxels) → unifocal / multifocal | Multifocal cases stress the metric; global vs lesion-wise Dice diverge here |
| **Peri-cavity confound** | min distance ET→RC (label-4), binned near_rc (≤5 mm) / far_rc / no_rc | The **direct 00533/02078 axis**: ET hugging the resection margin, where treatment-related enhancement confounds intensity localization |
| **Enhancement confound** *(optional, `--with-images`)* | 1 − overlap(brightest enhancement blob, ET), from `mapa_dif = norm(T1c) − norm(T1n)` | The exact Stage-3D mechanism that produced the irreducible failures, measured pool-wide as a per-case risk score |

The first three are computed from the seg mask alone (cheap). The fourth loads T1c+T1n
(~3× slower) and is the most thesis-relevant, because it quantifies — across the whole
population — how often "the brightest enhancement is non-tumor" actually occurs.

## Design decisions (and their knobs)

- **Characterize first, then sample.** You cannot pick a balanced sample without measuring
  every candidate. `characterize` writes one row per case; `sample` bins and draws. Re-run
  `sample` freely (cheap) with different seeds/sizes without recomputing measurements.
- **150 pool / 100 process.** `--pool-size 150` selects the manifest cohort; `--process-size
  100` marks the working subset (`process=1`) that the classical baseline + Stage 4 run on.
  The extra 50 are a reserve for later expansion or replacement.
- **Guaranteed minimums on hard strata.** `--min-absent / --min-small / --min-multifocal /
  --min-near-rc` force representation of failure-relevant cases so they aren't swamped by
  common easy cases. Remaining slots fill proportionally to available stratum sizes.
- **Stratified k-fold on the process set.** `--folds 5` assigns each process case a `fold`,
  stratified within strata. **This is what makes "use the priors on the 100" legitimate:**
  Phase-2 priors (shape model, ET-atlas, expected Betti) are built from labeled data, so
  they must be trained on training folds and evaluated on a held-out fold. Never build a
  prior on a case you then score — that inflates results.
- **Seeded throughout** (`--seed`), so the exact cohort, process subset, and folds are
  reproducible and recorded in the manifest header.

## How to run

```bash
# 0. Unzip the training data somewhere, e.g. D:\brats\TrainingData\
#    (each case is a folder <case_id>/ with -t1c/-t1n/-t2w/-t2f/-seg .nii.gz)

# 1. Characterize the full pool (seg-only, fast — do this first)
python stratify_brats.py characterize \
    --data-root "D:/brats/TrainingData" \
    --out pool_characterization.csv

#    OR with the enhancement-confound proxy (slower, loads T1c+T1n):
python stratify_brats.py characterize \
    --data-root "D:/brats/TrainingData" \
    --out pool_characterization.csv --with-images

# 2. Draw the stratified cohort + write the searchable manifest
python stratify_brats.py sample \
    --pool pool_characterization.csv \
    --manifest COHORT_MANIFEST.md \
    --pool-size 150 --process-size 100 --folds 5

# Outputs:
#   pool_characterization.csv   — every case measured (the full pool, searchable)
#   COHORT_MANIFEST.md          — the 150 selected, with strata + process flag + fold
#   COHORT_MANIFEST_selected.csv — machine-readable version of the manifest rows
```

## Searching / extending the cohort

- **Find a case:** `case_id` is the folder name in the unzipped data.
- **Search by property:** `pool_characterization.csv` has every case with all measured
  variables — filter it (e.g., all multifocal near_rc cases with ET < 500 mm³) to find or
  add cases beyond the 150.
- **Add cases:** re-run `sample` with a larger `--pool-size`, or append rows to the manifest
  keeping the same columns (populate the measured fields from the pool CSV).

## Caveats (state these in any writeup)

- Post-treatment resection deforms anatomy, so the peri-cavity and (Phase-2) atlas priors
  are softer here than in pre-treatment data.
- The enhancement-confound proxy is a heuristic risk score, not ground truth; it flags
  *candidates* for the confound failure, to be confirmed per case.
- Bins use pool-wide percentiles, so they shift if the pool changes — the manifest header
  records the exact p33/p66 used for that run.