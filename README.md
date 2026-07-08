# BraTS 2024 GLI ET Segmentation Study

This repository contains a reproducible classical medical-image segmentation
study for BraTS 2024 GLI enhancing tumor (ET, label 3). The project started as a
20-case classical/deformable baseline, then expanded to a stratified 100-case
post-treatment cohort and Phase 2 prior experiments.

## What Is Load-Bearing

- `run_all.py` is the single-command runner for preprocessing, segmentation,
  metrics, figures, checkpoints, and provenance.
- `configs/pipeline.yaml` defines the core method list, legacy method list,
  seed, guard thresholds, and preprocessing/segmentation parameters.
- `src/brats_pipeline/` contains the core pipeline. The deterministic Chan-Vese
  reset, global seed reset, and Stage 3A evidence guard are intentionally pinned.
- `phase2/metrics.py` is the locked Stage-4/Phase-2 metric implementation. It is
  the scoring surface for every Phase 2 comparison.
- `analysis/baseline/` contains frozen regression baselines and fixture data.
- `cohort/COHORT_MANIFEST.md` and `cohort/COHORT_MANIFEST_selected.csv` define
  the exact 150/100 stratified cohort.

Do not change any of the above casually. Any numerical change requires rerunning
the relevant baseline and audit gates.

## Directory Map

| path | purpose |
| --- | --- |
| `src/brats_pipeline/` | Core preprocessing, segmentation, metrics helpers, visualization. |
| `configs/` | YAML pipeline configuration. |
| `tests/` | Regression and determinism tests. |
| `analysis/` | Phase-1 audits, frozen baseline reports, Stage 3/4 diagnostics. |
| `cohort/` | Cohort stratification tool, spec, and manifests. |
| `phase2/` | Locked metric plus spatial, shape-proxy, cubical-PH, surface, and consolidation experiments. |
| `output/` | Default runner outputs when using the 20-case baseline layout. |

The gated BraTS training data directory is ignored by git and must not be
committed.

## Run The Core Pipeline

```powershell
.\.venv\Scripts\python.exe run_all.py --config configs\pipeline.yaml
```

Useful variants:

```powershell
# Existing cleaned volumes, no figures or Poisson reconstruction.
.\.venv\Scripts\python.exe run_all.py --config configs\pipeline.yaml --skip-clean --skip-viz --skip-poisson

# Run only process=1 cases from the selected cohort manifest.
.\.venv\Scripts\python.exe run_all.py --config configs\pipeline.yaml `
  --data-root BraTS2024-BraTS-GLI-TrainingData\training_data1_v2 `
  --cohort cohort\COHORT_MANIFEST_selected.csv --process-only `
  --output-root output --clean-root output\limpieza

# Include legacy/cut methods without deleting their code.
.\.venv\Scripts\python.exe run_all.py --config configs\pipeline.yaml --legacy
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_regression.py
.\.venv\Scripts\python.exe -m pytest tests\test_determinism.py
```

`tiempo_s` is reporting only and is excluded from deterministic assertions.

## Phase Summary

- Phase 1 / Stage 3: deterministic classical/deformable ET baseline on the
  original 20 cases; variational_spline retained as the best deformable method.
- Cohort build: stratified 150-case pool with 100 process cases, balanced folds,
  and explicit absent/small/medium/large ET strata.
- Stage 4: locked, case-type-aware metrics. ET-absent cases are scored by false
  positive burden instead of degenerate Dice.
- Phase 2: compares four regimes on the locked metric: intensity baseline,
  spatial/location atlas, shape-proxy geometry, and genuine cubical H0 persistent
  homology.

