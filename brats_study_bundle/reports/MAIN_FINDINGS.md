# Main Findings

## Abstract-Style Summary

This study evaluates classical and geometry-constrained enhancing-tumor (ET)
segmentation on a stratified post-treatment BraTS-2024 GLI cohort. The central
finding is negative but informative: methods and priors operating on the
enhancement scalar cannot reliably discriminate residual tumor from
treatment-related enhancement. The failure appears on two axes. First, baseline
methods hallucinate ET on ET-absent cases, with absent-case flood rates of
0.939-1.000 across the five core methods. Second, post-hoc priors improve parts
of the false-positive axis only by sacrificing detection, selecting operating
points post hoc, or leaving high residual flood.

The four tested regimes form an impossibility chain. Intensity fails because the
irreducible confounds are themselves strongly enhancing. Location fails because
those confounds are not spatial outliers in the learned ET-occurrence atlas.
Shape-proxy geometry separates components in isolation, but does not deploy as a
robust post-filter. Genuine cubical H0 persistent homology fails most directly:
confound components are more persistent than true ET on the enhancement scalar
(AUC true ET > confound = 0.126). Surface reconstruction confirms the voxel
metric rather than rescuing it: on the large stratum, lesion-wise Dice and
surface ASD are monotonically opposed across methods (Spearman rho = -1.0).

The conclusion is that the discriminating information needed for post-treatment
ET segmentation is absent from the scalar enhancement representation used by the
classical pipeline. The next method should integrate multimodal, spatial,
geometric, and topological information during prediction rather than applying
post-hoc correction to an already ambiguous intensity segmentation.

## Study Axes And Figures

The final paper figures are stored in `phase2/figures/`.

| figure | role |
| --- | --- |
| Figure 1, `01_impossibility_chain_four_regimes.png` | Four-regime separability diagnostic. |
| Figure 2, `02_growth_metric_method_trajectories.png` | Method trajectories across baseline, spatial, shape-proxy, and persistence regimes. |
| Figure 3, `03_best_vs_worst_variational_spline_vs_gmm2d.png` | Best versus worst method comparison on overlap, FP burden, and surface ASD. |
| Figure 4, `04_per_stratum_lesionwise_and_surface_asd.png` | Stratum-specific lesion-wise Dice and surface ASD. |
| Figure 5, `05_absent_case_flood_and_fp_volume.png` | Absent-case flood and false-positive burden. |
| Figure 6, `06_poisson_surface_renders_good_vs_irreducible.png` | Poisson surface renders for a good case and irreducible confounds. |
| Figure 7, `07_surface_vs_dice_large_stratum.png` | Surface-distance versus Dice agreement on the large stratum. |

Canonical data tables are in `phase2/final_data/`.

## Regime 1: Intensity Baseline

**Robust descriptive finding.** The five core classical methods all operate on
T1c intensity or T1c-T1n enhancement evidence. On ET-absent cases, none of the
baseline methods called the case empty. Flood rates were 0.939 for `otsu_T1c`,
0.970 for `gmm_T1c`, 0.939 for `sustraccion`, 1.000 for `gmm_2d`, and 0.970 for
`variational_spline`. Median false-positive volumes ranged from 13,935 voxels
for `otsu_T1c` to 94,636 voxels for `gmm_2d` (Figure 5;
`stage4_absent_fp_summary.csv`).

**Suggestive detection finding.** On large ET-present tumors, the best baseline
detection method was `variational_spline`, with lesion-wise Dice 0.176. This is
the highest large-stratum value in the baseline, but it remains low in absolute
terms and is based on n=17 large cases. The result should be presented as the
best observed detector in this cohort, not as a clinically competent detector
(`stage4_present_by_vol_bin.csv`).

**Mechanistic implication.** The two irreducible original-20 ROI failures
(`00533`, `02078`) had maximal enhancement support and among the highest mean
enhancement values in the audit. Thus an enhancement-family veto cannot
separate tumor from non-tumor enhancement in the hard cases. This is
mechanistically strong but based on small n, so it is **suggestive** rather than
confirmatory.

## Regime 2: Spatial / Location Prior

**Suggestive negative finding.** The ET-occurrence atlas was trained
leave-fold-out and applied to held-out cases. It reduced false positives for
some thresholds, but not in a way that preserved detection robustly. The usable
spatial operating point for `variational_spline` reduced absent flood to 0.818
while retaining 92.2% of the baseline large-stratum detection signal, but median
FP volume remained 18,518 voxels.

The key mechanistic diagnostic is that the known confound seeds were not atlas
outliers. Their atlas percentiles were high, approximately 91.9 and 86.9 among
nonzero atlas voxels. Therefore, population location cannot reliably suppress
the difficult treatment-related enhancement patterns. This is **anecdotal to
suggestive** because it rests on two irreducible cases and five held-out atlases,
but it explains why the spatial prior did not solve the false-positive axis
(Figure 1; `four_regime_discrimination_diagnostic.csv`).

## Regime 3: Shape-Proxy Geometry

**Robust component-probe, suggestive deployment finding.** Normalized shape
features such as compactness, isoperimetric quotient, normalized radius, and
erosion-fragmentation separated true ET components from absent-case false
positive components with high offline AUC (0.999). This component-level result
is strong for absent floods, but it does not automatically translate into a
deployable segmentation rule because components are clustered by case and the
operating threshold was explored on the same held-out sweep used for reporting.

The best soft shape-proxy behavior was false-positive reduction without complete
detection collapse. `variational_spline` reached absent flood 0.727 with median
FP 14,118 voxels and large-stratum Dice 0.163. `otsu_T1c` reached absent flood
0.788 with median FP 13,772 voxels and large-stratum Dice 0.165. The result is
**suggestive**, not robust: residual flood remains high, the median-FP gain for
`otsu_T1c` is small, and the operating point was post-hoc selected (Figure 2;
`p2_soft_shape_operating_points.csv`).

Terminology is important. This regime is not persistent homology. It is
shape-proxy geometry or topology-inspired component plausibility.

## Regime 4: Genuine Cubical Persistent Homology

**Robust diagnostic negative finding.** P3 is the only regime that computes
genuine cubical persistent homology, using GUDHI cubical H0 persistence on the
normalized enhancement map. The diagnostic is inverted: true ET components had
median normalized H0 persistence 0.583, while all-confound components had median
persistence 1.000. The AUC for true ET ranking above confound was 0.126.

This is the strongest evidence against pushing a Morse or persistence pipeline
on the same scalar enhancement field. A persistence count/rank prior did not
beat the soft shape-proxy result and did not rescue the worst methods. It tied
the baseline best detector at best and preserved large detection only by leaving
the false-positive problem substantially intact (Figure 1;
`p3_persistence_diagnostic_summary.csv`).

## Surface Reconstruction Confirmation

**Robust geometric confirmation.** Poisson reconstruction was first calibrated
against GT ET masks to establish an error floor. The large-stratum GT floor had
median HD95 2.56 mm and median ASD 1.13 mm. Prediction surfaces were much worse:
large-stratum median ASD ranged from 13.86 mm for `variational_spline` to 22.00
mm for `gmm_2d`, an order-of-magnitude increase over the GT reconstruction
floor. Thus poor surface fidelity is not explained by the reconstruction
procedure alone.

Surface error also tracked the voxel metric. Across the five methods on the
large stratum, lesion-wise Dice and median surface ASD had Spearman rho = -1.0.
The surface analysis therefore confirms the segmentation ranking rather than
revealing hidden geometric success (Figures 6 and 7;
`surface_prediction_fidelity_by_method_stratum.csv`,
`surface_dice_relationship_by_stratum.csv`).

Native post-treatment ET geometry is itself difficult: GT reconstruction error
increased with multifocality and irregular, low-isoperimetric geometry. This
supports the interpretation that post-treatment ET is not well modeled as a
single compact, spherical enhancing object.

## Growth Metric Result

The growth table compares each method across baseline, P1, P2b, and P3 on the
same locked metric axes: absent flood rate, absent median FP volume, and
large-stratum lesion-wise Dice. The trajectories are not monotone improvements.
P1 can eliminate flood for flood-prone methods only at degenerate detection
settings. P2b offers the best exploratory FP/detection tradeoff, but does not
remove residual flood. P3 does not beat P2b and does not improve the baseline
best detector.

The worst false-positive method at baseline, `gmm_2d`, improved numerically
under P3 from flood 1.000 to 0.909 and median FP 94,636 to 72,438 voxels, but
large-stratum lesion-wise Dice remained below the 90% detection gate. The worst
detectors therefore did not cross into a usable regime. The best detector
remained `variational_spline` at baseline/P3 large-stratum Dice 0.176 (Figure 2;
`growth_metric_table.csv`, `growth_metric_improvement_from_baseline.csv`).

## Central Conclusion

The four regimes converge on the same interpretation. Post-hoc correction of
classical intensity segmentation cannot reliably distinguish post-treatment
tumor from treatment-related enhancement when both are represented by the same
scalar enhancement signal. Intensity, location, shape proxies, and genuine
cubical H0 persistence each fail in a different way, and their failures are
mutually explanatory rather than accidental. The next system must integrate
discriminating evidence during prediction, using multimodal image context,
learned spatial priors, and topology-aware objectives, rather than attempting to
repair ambiguous masks after segmentation.

