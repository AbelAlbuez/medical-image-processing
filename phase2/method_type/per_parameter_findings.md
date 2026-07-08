# Per-Parameter Behavior Findings

Source tables:

- `phase2/method_type/per_parameter_behavior.csv`
- `phase2/method_type/per_parameter_usable_windows.csv`

## P1 Spatial Sweep

The P1 table records the full threshold curves from `p1_spatial_sweep_summary.csv` for pre-filter, post-filter, and MAP modes. The headline read remains:

- MAP can drive flood to zero only by collapsing detection.
- Pre-filter has a sharp flood/detection cliff.
- Post-filter on `variational_spline` has the only modest P1 operating point: flood `0.818`, median FP `18,518`, large retention `92.2%`.

## P2b Soft Shape Sweep

Clean usable window definition:

- absent flood below Otsu baseline `0.939`
- median absent FP below Otsu baseline `13,935`
- large detection retention at least `90%`

Methods with a clean usable window:

otsu_T1c

Loose usable window definition:

- absent flood below the method's own baseline
- large detection retention at least `90%`

Methods with a loose usable window:

otsu_T1c, variational_spline

## Window Details

| method | type | clean window | loose window | first flood drop vs own | first large drop below 90% |
|---|---|---|---:|---:|
| `otsu_T1c` | intensity/statistical | yes (0.001-0.010) | yes (0.001-0.010) | 0.001 | 0.020 |
| `gmm_T1c` | intensity/statistical | no | no | 0.020 | 0.020 |
| `gmm_2d` | intensity/statistical | no | no | 0.001 | 0.000 |
| `sustraccion` | intensity/statistical | no | no | 0.001 | 0.000 |
| `variational_spline` | deformable/region-based | no | yes (0.001-0.020) | 0.001 | 0.050 |

## Type Interaction

The hypothesis was that the deformable method would have the widest usable shape-prior window because its true-tumor predictions are already compact.

The data only partly support this:

- `variational_spline` does have a loose usable window and the best flood reduction while preserving large detection: threshold `0.020`, flood `0.727`, large retention `92.6%`.
- But it misses the clean window because median FP remains slightly worse than Otsu baseline (`14,118` vs `13,935`).
- `otsu_T1c`, an intensity/statistical method, is the only method satisfying the stricter clean window, because its baseline median FP is already low and the soft shape threshold trims just enough FP while preserving large detection.
- `gmm_T1c` preserves detection at threshold `0`, but has no useful FP reduction window before detection degrades.
- `gmm_2d` and `sustraccion` have no meaningful clean shape-prior window.

Conclusion:

> Shape prior response is not determined by construction type alone. Deformable masks are more shape-compatible under aggressive filtering, but the clinically useful soft operating point is controlled by the joint distribution of baseline FP volume and true-tumor component scores.
