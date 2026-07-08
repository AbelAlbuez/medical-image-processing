# Method Type x Stage Findings

Source table: `phase2/method_type/method_type_stage_matrix.csv`

## Construction-Type Classification

- Intensity/statistical: `otsu_T1c`, `gmm_T1c`, `gmm_2d`, `sustraccion`, plus cut method `rango_doble`.
- Deformable/region-based: `variational_spline`, plus cut method `level_set`.
- Deformable/control-point: cut methods `bspline`, `spline`.
- Front-propagation: cut method `fast_marching`.

## Does Construction Type Predict Behavior?

Partly, but not cleanly enough to use as a deterministic rule.

### Stage 4 Baseline

The intensity/statistical core methods do flood strongly on ET-absent cases:

- `otsu_T1c`: flood `0.939`, median FP `13935` vox.
- `gmm_T1c`: flood `0.970`, median FP `18837` vox.
- `gmm_2d`: flood `1.000`, median FP `94636` vox.
- `sustraccion`: flood `0.939`, median FP `76915` vox.

The deformable/region-based `variational_spline` is more restrained by maximum FP than the worst intensity methods, but it still floods:

- `variational_spline`: flood `0.970`, median FP `18837` vox, max FP `278959` vox.

For present tumors, `variational_spline` has the best large-stratum lesion-wise score (`0.176`), consistent with the idea that the region-based method better preserves a main mass. But it does not solve small or multifocal detection.

### Phase 1 Oracle/Redundancy

Construction type alone did not decide keep/cut. The five core methods all had positive leave-one-out oracle deltas, while the legacy methods had zero delta. This supports pruning by measured unique value, not by type label alone.

### P1 Spatial Prior

P1's useful operating point was reported only for `variational_spline` post-filter. It reduced flood to `0.818` at `92.2%` large-detection retention but worsened median FP relative to Otsu. This does not establish a general type effect.

### P2a Hard Shape Filter

P2a shows the clearest type interaction:

- `variational_spline` retained `63.2%` of large detection under hard shape filtering.
- `otsu_T1c` retained only `7.4%`.
- `gmm_T1c` retained `8.1%`.

This supports the hypothesis that deformable predictions are more shape-compatible than raw intensity-threshold predictions. However, P2a still failed the 90% detection-preservation gate.

### P2b Soft Shape Prior

P2b complicates the type story:

- The clean-bar exploratory point is `otsu_T1c` at threshold `0.010`, flood `0.788`, median FP `13,772`, large retention `93.6%`.
- `variational_spline` has a stronger flood reduction (`0.727`) and retains `92.6%` large detection, but median FP is `14,118`, slightly worse than Otsu's baseline.

Interpretation: type predicts baseline morphology and hard-filter survivability, but the best soft operating point depends on the starting method's FP distribution. The strongest paper-safe statement is:

> Intensity/statistical methods tend to detect by broad intensity capture and flood ET-absent cases; the deformable method is more compact and better aligned with shape filtering, but it still floods and misses satellites. Construction type explains tendencies, not outcomes.
