# P2b Headline Claim Audit Addendum

## Classification

The P2b soft-shape result is **suggestive**, not robust.

Observed headline operating point:

- Method: `otsu_T1c`
- Soft shape-score threshold: `0.010`
- Absent-case flood rate: `0.939 -> 0.788`
- Flood-rate delta vs Otsu baseline: `-0.152`, bootstrap CI `[-0.273, -0.030]`
- Median absent FP volume: `13,935 -> 13,772`, delta `-163 vox`
- Large-stratum lesion-wise Dice: `0.165`
- Large-stratum detection retention: `93.6%`

The direction is favorable, but the evidence is thin for a headline robustness claim:

- The flood-rate CI barely clears zero on the upper end (`-0.030`).
- The median-FP improvement is only `163 vox` on predictions around `14k vox`.
- Residual absent-case flood remains high at `78.8%`; this does not solve false positives.
- The result comes from a single 100-case cohort.

## Operating-Point Provenance

The shape plausibility **model scores** were learned correctly on training folds only:

- Fold-specific logistic models were fit from normalized features only.
- Features: `log10(isoperimetric)`, `radius/bbox_diag`, `radius/equivalent_sphere`, `log1p(eroded_subcomponents)`.
- No raw volume or raw radius was used.

However, the reported P2b operating point at threshold `0.010` was **not itself fixed from training folds**.

In `phase2/p2_soft_shape_sweep/soft_sweep.py`, thresholds are swept on held-out predictions via `SCORE_THRESHOLDS`, and the reported operating point is selected from held-out sweep summaries as:

`lowest_fp_while_large_detection_ge_90pct`

Therefore the `0.010` threshold is a **post-hoc held-out operating-point selection**, not a train-fold-frozen threshold. The clean-bar-cleared claim must be framed as an exploratory/suggestive sweep result, not as a fully locked validation result.

## Honest Headline

The honest headline is:

**Soft shape is more promising than spatial location as a false-positive prior, but it is not clinically sufficient.**

More specifically:

- P2b soft shape improves the FP/detection tradeoff compared with P1 spatial filtering.
- It reduces flood rate more than P1 at comparable large-tumor detection.
- Unlike P1, it slightly improves median FP volume vs Otsu at the selected point.
- But residual flood is still `78.8%`, and the selected threshold was chosen from the held-out sweep.

Recommended wording:

> Shape-normalized component plausibility produced the best exploratory FP tradeoff observed in Phase 2, reducing absent-case flood from 93.9% to 78.8% while retaining 93.6% of large-tumor detection. Because the operating threshold was selected post hoc from the held-out sweep and residual flood remained high, this is suggestive evidence that shape is a stronger FP prior than location, not a clinically sufficient solution.
