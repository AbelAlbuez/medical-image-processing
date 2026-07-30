# Caveats And Gaps

This document states the limitations that should appear explicitly in the paper
or thesis chapter. The goal is not to weaken the result, but to keep the claims
review-proof.

## 1. P2b Operating Point Was Post-Hoc

**What it is.** The P2b soft shape scores were learned on training folds, but
the reported operating threshold was selected by reading the held-out sweep. The
`otsu_T1c` threshold 0.010 and the `variational_spline` loose operating point
are exploratory.

**Why it matters.** The false-positive drop can be real in this cohort while
still being optimistically selected. It must not be described as a locked
validation result.

**How to address it.** Freeze the method and threshold before scoring an
external cohort or a newly held-out split. Report P2b currently as
**SUGGESTIVE**.

## 2. Tiny-n Mechanistic Claims

**What it is.** The large-unifocal subset has n=2. The peri-cavity shape probe
has n=4 false-positive components. The irreducible mechanism analysis centers
on two cases, `00533` and `02078`.

**Why it matters.** These are useful mechanism illustrations, but a reviewer
will reject them as headline statistical claims.

**How to address it.** Keep these cases in a qualitative or mechanistic
subsection. Phrase as "consistent with" and "illustrates." Do not report them as
primary evidence.

## 3. Single-Cohort Evaluation

**What it is.** The Phase-2 conclusions are based on one stratified 100-case
BraTS-2024 GLI cohort.

**Why it matters.** The negative false-positive finding is strong in this
cohort, but generalization to other post-treatment cohorts, scanners, or label
conventions is not externally validated.

**How to address it.** Add an external validation cohort or repeat the locked
protocol on an untouched BraTS split. Until then, state "in this stratified
cohort."

## 4. Determinism Scope

**What it is.** Determinism is proven for runner masks from cleaned inputs and
for regression fixtures. End-to-end bit-identical determinism from raw NIfTI
through N4 bias correction is not fully proven in the visible tests.

**Why it matters.** N4 and image-cleaning libraries may involve platform,
threading, or ITK-version behavior. Overclaiming end-to-end reproducibility
would invite replication criticism.

**How to address it.** Either add a raw-input determinism test including
cleaning for at least one case, or scope the claim to "cleaned-input
segmentation determinism."

## 5. Shape Proxy Versus Persistent Homology Terminology

**What it is.** P2 uses compactness, isoperimetric, normalized-radius, and
erosion-fragmentation proxies. These are topology-inspired geometry features,
not persistent homology. P3 is the first genuine cubical persistent homology
computation.

**Why it matters.** A mathematical reviewer will notice if proxy geometry is
called PH or Morse theory.

**How to address it.** Use precise labels:

- R3: shape-proxy geometry or topology-inspired proxies.
- R4: genuine GUDHI cubical H0 persistent homology.

## 6. Multiple-Comparisons Exposure

**What it is.** P1 swept about 100 configurations, P2b swept about 60
configurations, and multiple paired tests were inspected.

**Why it matters.** A p-value or CI from the best-looking configuration is not
confirmatory without correction or locked validation.

**How to address it.** Label sweep results exploratory. Use confidence intervals
for descriptive effect sizes. Validate only pre-specified settings.

## 7. Original-20 Overlap Cases

**What it is.** Three original-20 development cases overlap the 100-case process
cohort: `BraTS-GLI-02086-100`, `BraTS-GLI-02143-100`, and
`BraTS-GLI-02151-100`.

**Why it matters.** Stage-3 guard tuning and regression checks used the
original-20 set. Even though the overlap is small, it is a leakage exposure.

**How to address it.** Report a sensitivity analysis excluding those three
cases. If the conclusion remains unchanged, the concern becomes minor.

## 8. Baseline Winner Selection Is Descriptive

**What it is.** The "best baseline" is selected on the same 100-case cohort used
for reporting.

**Why it matters.** This is acceptable for descriptive method comparison, but
not for a claim that the selected method is generally optimal.

**How to address it.** Say "best observed baseline in this cohort" and compare
priors against per-axis baselines rather than a single combined score.

## 9. Surface Reconstruction Is An Evaluation Axis, Not A Segmentation Fix

**What it is.** Poisson surfaces are reconstructed from masks to evaluate
geometry and failure modes.

**Why it matters.** Reconstruction error has its own floor and can worsen for
multifocal or irregular ET. Surface figures should not imply that Poisson
reconstruction improves the masks.

**How to address it.** Present the GT reconstruction floor first, then compare
prediction surfaces against that floor.

## 10. Bibliography Verification Remains Needed

**What it is.** The audit identified prior-art anchors but did not complete a
paper-ready bibliography for all requested items.

**Why it matters.** Biomedical-mathematics reviewers will expect exact claims
and citations for train-free cubical PH, atlas priors, and tumor-shape topology.

**How to address it.** Build a verified `.bib` file and check that every
statement in the introduction maps to a cited source.

