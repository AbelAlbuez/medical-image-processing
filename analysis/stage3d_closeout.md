# Stage 3D closeout: ROI-init failures 00533 and 02078

No masks, baselines, or segmentation parameters were changed. 3B remains cut.

## Decision

`BraTS-GLI-00533-100` and `BraTS-GLI-02078-100` are documented limits of the current enhancement-driven classical ROI initialization. They should not be repaired by another threshold/veto pass.

The failure mechanism is not weak enhancement. It is the opposite: the GMM seed locks onto the brightest enhancing non-tumor structure in the case. In `analysis/stage3d_gmm_seed_support.csv`, both failures have maximal support:

| case | GMM support frac above brain p90 | GMM mean mapa_dif | rank by mean mapa_dif | variational_spline Dice |
| --- | ---: | ---: | ---: | ---: |
| `BraTS-GLI-02078-100` | 1.0000 | 0.6112 | 1 / 20 | 0.0000 |
| `BraTS-GLI-00533-100` | 1.0000 | 0.5329 | 2 / 20 | 0.0000 |

Because these two sit at the top, not the bottom, no enhancement-support veto can separate them from the 18 non-target cases. Good cases such as `02151`, `02143`, and `00020` also have very high GMM enhancement support, so forcing a threshold would create avoidable regressions.

## Mechanism

Visual inspection overlays are saved in `analysis/stage3d_irreducible_overlays/`.

| case | mislocalization target | GMM centroid z/y/x | brain-normalized z/y/x | distance to GT centroid | relation to RC | interpretation |
| --- | --- | --- | --- | ---: | --- | --- |
| `BraTS-GLI-00533-100` | resection-margin / cavity-adjacent non-tumor enhancement | `87.04;52.18;87.33` | `0.53;0.19;0.49` | 38.7 vox | 1.0 voxel from RC | central/anterior-periventricular bright enhancement adjacent to the resection cavity, separate from lateral ET |
| `BraTS-GLI-02078-100` | midline vascular/dural-like non-tumor enhancement near resection region | `74.17;52.27;91.15` | `0.47;0.20;0.50` | 67.6 vox | 6.2 vox from RC | midline/anterior-inferior bright linear enhancement, separate from lateral ET and RC |

Supporting file: `analysis/stage3d_irreducible_mechanism.csv`.

## Relation to BraTS 2024 GLI Post-Treatment

This matches the post-treatment difficulty described in the local thesis/report material: BraTS 2024 GLI is explicitly a post-treatment setting and introduces resection cavity (`label 4`) as a distinct class. In this setting, enhancement-only classical methods are vulnerable to non-tumor enhancement, cavity-adjacent enhancement, vascular/dural enhancement, and treatment-related change. These structures can be brighter than the actual ET in T1c/T1n difference space, so an intensity-only selector can be confidently wrong.

For these two cases, the measured evidence supports that ceiling:

- The GMM seed is disjoint from ET: Dice 0.0000 in both cases.
- The GMM seed is highly enhancing: support fraction 1.0000 in both cases.
- The GMM seed mean enhancement exceeds the ET mean enhancement:
  - `00533`: GMM 0.5329 vs GT ET 0.1171.
  - `02078`: GMM 0.6112 vs GT ET 0.3602.
- The enhancement fallback intersects ET, but is broad and low-Dice, so swapping seeds without a stronger prior would move many other cases.

## Closeout

These two failures are irreducible for the current classical, intensity-led ROI initializer. Fixing them safely would require additional information outside this baseline: spatial priors, anatomy-aware shape constraints, lesion-wise context, or learning-based models trained to distinguish tumor enhancement from post-treatment non-tumor enhancement.

No seed-logic change is recommended. Deformable/ROI segmentation-quality work is closed. The next stage is Stage 4: lesion-wise metrics.
