# Paper Outline

## Working Title

Limits of Classical Enhancement-Driven ET Segmentation in Post-Treatment
BraTS-2024 GLI: A Four-Regime Negative Result With Surface and Topological
Diagnostics

## 1. Abstract

**Claim:** Classical enhancement-driven ET segmentation fails primarily through
false-positive burden and does not become reliable after spatial, shape-proxy,
or cubical-persistence post-processing.

**Robustness:** Mixed. Baseline FP failure is **ROBUST**; P2b improvement is
**SUGGESTIVE**; peri-cavity examples are **ANECDOTAL**.

**Figures/Tables:** Figures 1, 2, 5; `four_regime_summary.csv`.

## 2. Introduction

### 2.1 Clinical and methodological problem

Post-treatment enhancing abnormalities can be tumor, treatment effect,
resection-margin enhancement, vascular enhancement, or other non-tumor signal.
Classical intensity segmentation assumes the enhancement scalar is sufficient.

**Claim class:** Background plus study motivation.

### 2.2 Why geometry/topology is tempting

Topology and shape priors are attractive because ET is often expected to be
compact or blob-like. However, post-treatment ET can be multifocal, shell-like,
or irregular.

**Claim class:** Requires bibliography hardening.

### 2.3 Contribution

This paper provides a controlled negative result across four regimes:

1. Intensity baseline.
2. Spatial/location prior.
3. Shape-proxy geometry.
4. Genuine cubical H0 persistent homology.

**Claim class:** **ROBUST** as a description of the experiments.

## 3. Data And Cohort

### 3.1 BraTS-2024 GLI cohort and label convention

Use ET label 3. Describe the stratified 150-case pool and 100 process cases.
Report absent/small/medium/large composition.

**Claim class:** **ROBUST descriptive**.

**Table:** `cohort_manifest_selected.csv`.

### 3.2 Metric lock

Explain why pooled global Dice is not a fair summary for this cohort. Define the
locked metric:

- ET-present: FP-aware lesion-wise Dice, global Dice for continuity, HD95.
- ET-absent: false-positive volume and flood rate.

**Claim class:** **ROBUST metric rationale**, with caveat that metric definition
was refined during analysis.

**Table:** `stage4_case_metrics.csv`.

## 4. Methods

### 4.1 Classical intensity baseline

Describe `otsu_T1c`, `gmm_T1c`, `sustraccion`, `gmm_2d`, and
`variational_spline`.

**Claim class:** **ROBUST implementation description**.

### 4.2 Spatial ET-occurrence atlas

Describe leave-fold atlas construction and pre/post/MAP modes.

**Claim class:** **ROBUST for atlas construction**, **SUGGESTIVE for operating
point selection**.

### 4.3 Shape-proxy geometry

Define compactness, isoperimetric quotient, normalized radius, and erosion
fragmentation. State explicitly that this is not persistent homology.

**Claim class:** **ROBUST component calculation**, **SUGGESTIVE deployment**.

### 4.4 Cubical persistent homology

Describe GUDHI cubical H0 persistence on normalized enhancement-map crops and
the count/rank prior.

**Claim class:** **ROBUST diagnostic**.

### 4.5 Surface reconstruction

Describe Open3D Poisson reconstruction, GT reconstruction floor, and surface
metrics: HD95, ASD, Chamfer.

**Claim class:** **ROBUST evaluation axis**.

## 5. Results

### 5.1 Baseline false-positive burden

All five methods predicted nonempty ET on all 33 absent cases; flood rates were
0.939-1.000.

**Claim class:** **ROBUST descriptive**.

**Figure/Table:** Figure 5; `stage4_absent_fp_summary.csv`.

### 5.2 Present-case detection remains low

Large-stratum detection is the least noisy present-case signal. Best observed
large lesion-wise Dice was 0.176 for `variational_spline`.

**Claim class:** **SUGGESTIVE** because n=17.

**Figure/Table:** Figure 4; `stage4_present_by_vol_bin.csv`.

### 5.3 Four-regime impossibility chain

Intensity fails, location fails, shape-proxy geometry is only partially useful,
and cubical H0 persistence is inverted.

**Claim class:** R1 absent FP **ROBUST**; R2 hard-confound mechanism
**ANECDOTAL/SUGGESTIVE**; R3 component probe **ROBUST/SUGGESTIVE**; R4 inversion
**ROBUST**.

**Figure/Table:** Figure 1; `four_regime_discrimination_diagnostic.csv`.

### 5.4 Growth metric

Priors do not rescue the worst methods. Some FP reductions are degenerate or
post-hoc. Detection does not materially improve.

**Claim class:** **ROBUST negative trend**, with P2b positive point
**SUGGESTIVE**.

**Figure/Table:** Figure 2; `growth_metric_table.csv`,
`growth_metric_improvement_from_baseline.csv`.

### 5.5 Surface reconstruction confirms ranking

Prediction surface ASD is far above the GT reconstruction floor. On the large
stratum, surface ASD and lesion-wise Dice agree monotonically across methods
(Spearman rho = -1.0).

**Claim class:** **ROBUST geometric confirmation**, with n=5 method-level
correlation.

**Figure/Table:** Figures 6 and 7;
`surface_prediction_fidelity_by_method_stratum.csv`.

## 6. Discussion

### 6.1 Interpretation

The scalar enhancement field lacks the information needed to discriminate tumor
from treatment-related enhancement. Post-hoc geometric correction cannot create
that missing information.

**Claim class:** **ROBUST synthesis** if caveated to this cohort and these
methods.

### 6.2 Why integrated learning is required

The next method should combine multimodal image evidence, learned spatial
context, topology-aware losses, absence calibration, and surface/boundary-aware
evaluation.

**Claim class:** Proposed direction, not evaluated here.

### 6.3 Relation to topology literature

Position P3 relative to train-free cubical PH and shape-topology methods. State
that this study found raw-enhancement H0 persistence inverted for hard
post-treatment confounds.

**Claim class:** Requires bibliography verification.

## 7. Limitations

Include:

- P2b operating point selected post hoc.
- Single 100-case cohort.
- n=2 and n=4 mechanistic subsets.
- Original-20 overlap cases.
- Multiple sweeps and tests.
- Cleaned-input determinism scope.
- R3 proxy terminology.

**Claim class:** Limitations.

**Table:** `CAVEATS_AND_GAPS.md`.

## 8. Conclusion

Classical enhancement-driven segmentation and post-hoc priors fail on
post-treatment ET because the confound is not separable in the enhancement
scalar. The result motivates integrated learned segmentation with multimodal and
topology-aware constraints.

**Claim class:** **ROBUST as a scoped negative result**, **not** a claim about
all possible learned methods.

