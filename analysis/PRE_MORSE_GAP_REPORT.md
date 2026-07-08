# Pre-Morse Hardening Gap Report

Branch: `phase2`  
Scope: diagnostic audit only; no new methods and no experiment reruns.  
Primary artifacts audited: Stage 4 metric suite, P1 spatial atlas sweep, P2a hard shape filter, P2b soft shape sweep, cohort manifests, runner tests/provenance.

## Executive Verdict

The project has a coherent and valuable story, but the paper must be written conservatively:

- **Robust:** the cohort composition and baseline failure mode are well supported. Global Dice is unfair on this post-treatment cohort; absent/small ET cases dominate the near-zero pooled result; classical methods hallucinate ET on ET-absent cases.
- **Suggestive:** P1 spatial prior and P2b soft-shape prior are exploratory tradeoff analyses. P2b is the best Phase-2 result, but its operating threshold was selected post hoc from the held-out sweep, so it is **not** a locked-validation claim.
- **Anecdotal:** peri-cavity mechanism claims for `00533/02078`, n=4 peri-cavity FP components, and the n=2 large-unifocal "fairest case" cannot be headline claims.
- **Math rigor gap:** current P2 is **not cubical persistent homology**. It uses normalized shape/topology proxies: compactness, isoperimetric quotient, normalized inscribed-radius ratios, and erosion fragmentation. Calling these "persistent homology" would be mathematically overclaimed.
- **Reproducibility gap:** runner determinism is strongly shown from cleaned inputs and for runner masks, but an end-to-end raw-data/N4 bit-identical proof is not established in the visible tests.

The safest paper headline is:

> On a stratified post-treatment BraTS-2024 GLI cohort, classical ET segmentation is dominated by false-positive burden on ET-absent and small-ET cases. Shape-normalized component plausibility is a more promising false-positive prior than spatial location, but both remain insufficient for clinical deployment; P2b is an exploratory/suggestive result requiring locked external validation.

## A. Statistical Power And Claim Support

### Claims Inventory

| Claim | Artifact | n | CI/test | Classification | How to state |
|---|---|---:|---|---|---|
| Cohort contains substantial ET-absent stratum: 33/100 process cases absent; present split small 31, medium 19, large 17 | `cohort/COHORT_MANIFEST_selected.csv`, Stage 4 tables | 100 | Descriptive; no CI needed | **Robust descriptive** | Safe as cohort composition, not population prevalence unless framed as sampled cohort |
| Pooled global Dice is misleading because 63/79 near-zero cases are absent/small | `analysis/pre_stage4_resolution/*`, Stage 4 tables | 100 | Descriptive | **Robust** | Safe as motivation for stratified metrics |
| Baseline present-case lesion-wise scores are low across all methods; best all-present is `variational_spline` 0.054 | `stage4_method_ranking_detection_vs_fp.csv` | 67 present | No CI in table | **Suggestive to robust descriptive** | Safe as observed cohort result; add CIs before journal submission |
| Large-stratum detection is the only non-noise signal; best is `variational_spline` 0.176 lesion-wise, n=17 | `stage4_present_by_vol_bin.csv` | 17 | No CI | **Suggestive** | Do not overstate; n<20 and metric remains low |
| Medium detection best `variational_spline` 0.023, n=19 | `stage4_present_by_vol_bin.csv` | 19 | No CI | **Suggestive/noise-floor** | Use as context, not headline |
| Small detection best `sustraccion` 0.003, n=31 | `stage4_present_by_vol_bin.csv` | 31 | No CI | **Robust failure signal** | Robust that methods fail; not meaningful as ranking |
| Absent-case FP burden: all methods nonempty on 33/33 absent; flood rates 0.939-1.000 | `stage4_absent_fp_summary.csv` | 33 absent | No CI | **Robust descriptive** | Safe; add binomial CIs in manuscript |
| Large-unifocal "fairest case" numbers, e.g. `gmm_T1c` 0.431, `variational_spline` 0.427 | `large_unifocal_subset_by_method.csv` | **2** | No CI | **Anecdotal** | Must not be headline; phrase as sanity illustration only |
| P1 spatial post-filter at matched detection: flood 0.818, median FP 18,518, large retention 92.2% | `p1_fixed_detection_cost_operating_points.csv` | 33 absent, 17 large | Bootstrap CIs partly in P1 key table, selected from sweep | **Suggestive exploratory** | Spatial prior did not cleanly solve FP; avoid "validated improvement" |
| P1 atlas seed values at `00533/02078` are low absolute but high percentile among nonzero atlas voxels | `p1_seed_percentile_summary.csv` | 2 cases | No formal test | **Anecdotal/mechanistic** | Use to explain why atlas veto is weak, not as general claim |
| Pre-P2 shape separability: true ET vs absent FP compactness AUC 0.999 | `p2_shape_separation_tests.csv` | 180 true components vs 159 absent FP components | Mann-Whitney p tiny | **Suggestive, not robust headline** | Strong component-level signal, but components are clustered by case; independence assumption is weak |
| Peri-cavity shape separability AUC 1.000 | `p2_shape_separation_tests.csv` | 180 true vs **4** peri-cavity FP components | p shown, but n=4 FP | **Anecdotal** | Corroborating only; cannot headline |
| P2a hard filter crushes FP but fails detection preservation | `p2_shape_key_comparison_table.csv` | 100 cases | CIs/tests | **Robust enough for negative result** | Safe: hard binary filter is too brittle |
| P2b soft shape: Otsu threshold 0.010 gives flood 0.939->0.788, CI [-0.273,-0.030], median FP -163 vox, large retention 93.6% | `p2_soft_shape_operating_points.csv`, `p2b_headline_audit_addendum.csv` | 33 absent, 17 large | Bootstrap CI and paired tests | **Suggestive, not robust** | Best exploratory FP tradeoff; post-hoc operating point, residual flood 79% |
| Shape > location as FP prior | P1 vs P2b comparison | same cohort | Comparative, post-hoc | **Suggestive** | Safe if phrased as exploratory within-cohort comparison |

### Mandatory Downgrades

- **n=2 large-unifocal subset:** anecdotal. Use as an illustrative sanity check only.
- **n=4 peri-cavity AUC 1.000:** anecdotal. State "consistent with" shape separability, never "proves."
- **Any stratum with <10 cases:** no overall vol_bin is <10 in the 100 process set, but fold-level held-out bins are small: large is 3-4 per fold, medium 3-4, absent 6-7, small 6-7. Fold-wise prior behavior must not be overinterpreted.
- **Any AUC/rate on <20 samples:** large stratum n=17 and medium n=19 are suggestive. The large-stratum detection preservation gate is useful operationally but not high-powered.

### Multiple-Comparisons Exposure

Observed sweep/config load:

- P1 spatial sweep: **100 configs**.
- P2b soft shape sweep: **60 configs** (`12 thresholds x 5 methods`).
- P2a hard shape: 5 method-level operating rows after train-fold thresholding.
- P2 pre-probe separability tests: multiple metrics across absent FP and peri-cavity FP comparisons.
- P2b operating table includes multiple paired tests per method: flood, FP volume, and per-stratum detection deltas.

Implication:

- Any isolated p-value near 0.05 is weak after family-wise or FDR correction.
- P2b flood p/CI is directionally useful, but the upper CI is close to zero and the operating point was selected post hoc. Treat as **suggestive**.
- Negative findings with large obvious effects, e.g. hard P2a detection collapse, are more credible than small positive gains.

## B. Leakage And Evaluation Integrity

### Tuned Quantity Trace

| Quantity | Chosen on | Evaluated on | Leak status | Comment |
|---|---|---|---|---|
| Stage 4 corrected metric definition | Developed during 100-case analysis | Same 100 cases | **Method-definition leak risk** | Acceptable if framed as metric correction/audit, but final paper should freeze metric before final result table |
| Cohort stratification bins | Full 1350 pool characterization | 100 selected cohort | Low leak | Uses labels to sample strata; acceptable for stratified evaluation, not for model training |
| Per-axis baseline winner selection | 100-case Stage 4 results | Same 100 cases | **Selection-on-test** | Fine for descriptive "best observed baseline," not for confirmatory claim |
| 3A guard thresholds `0.90/0.90/2.50` | Original 20-case development | Original 20 and 100 cohort | **Partial leak into cohort** | 3/100 process cases overlap original 20 |
| P1 atlas construction | Other folds only | Held-out fold | Clean for atlas maps | Verified in `phase2/p1_spatial_atlas/build_atlas.py`: holdout fold uses the other 4 folds |
| P1 operating thresholds | Held-out sweep | Same held-out metrics | **Post-hoc operating-point selection** | Exploratory tradeoff, not locked validation |
| P2a shape model | Training folds only | Held-out folds | Clean for model score | Fold models in `p2_shape_thresholds_by_fold.csv` |
| P2a hard thresholds | Training folds only | Held-out folds | Clean for hard P2a | Train policy: max balanced accuracy with TPR >= 0.95 |
| P2b shape scores | Training folds only | Held-out folds | Clean for scores | Same fold-specific models |
| P2b soft threshold `0.010` | **Held-out sweep** | Same held-out sweep | **Leak/post-hoc** | Must not be called locked validation |

### P2b Addendum Finding

The P2b component shape scores were learned on training folds only. However, the reported operating point `otsu_T1c`, threshold `0.010`, was selected from the held-out sweep by the rule:

`lowest_fp_while_large_detection_ge_90pct`

Therefore:

- "Clean bar cleared" is **post-hoc selected**.
- The result must be framed as **exploratory/suggestive**.
- A locked validation requires choosing `method=otsu_T1c`, `threshold=0.010` before evaluating a new cohort or an untouched fold split.

### Original-20 Overlap Leak

Original 20 baseline cases overlap the 100 process cohort in 3 cases:

- `BraTS-GLI-02086-100`
- `BraTS-GLI-02143-100`
- `BraTS-GLI-02151-100`

Leak exposure:

- 3/100 cohort cases.
- 3/67 ET-present cases.
- These cases were involved in earlier guard/runner validation work. They should be flagged in any sensitivity analysis; ideally report results with and without these 3 cases.

### Held-Out Discipline Verification

- P1 atlas: clean at atlas-construction level. Each fold atlas averages ET masks from the other 4 folds only.
- P2a: clean at model and threshold level. Fold-specific thresholds were learned from training components only.
- P2b: clean at shape-score model level, **not clean at operating-point selection level**.

## C. Mathematical Rigor And Prior Art

### Computed vs Approximated Signals

Current P2 does **not** compute cubical persistent homology. It computes shape/topology proxies:

- `isoperimetric_quotient = 36*pi*V^2 / S^3`, where `V` is component voxel volume and `S` is exposed voxel-face surface proxy.
- `compactness = V / S^(3/2)`, same information up to scaling/monotonicity.
- `radius_over_bbox_diag = r_in / diag(B)`, where `r_in` is the largest inscribed-sphere radius proxy and `diag(B)` is bounding-box diagonal.
- `radius_over_equiv_sphere = r_in / ((3V)/(4*pi))^(1/3)`, normalized sphere signal.
- `eroded_subcomponents = number of connected components after 1-voxel binary erosion`, a fragmentation/branching proxy.

Terminology rule:

- Safe: "shape-normalized component plausibility," "compactness proxy," "topology-inspired proxy," "erosion-fragmentation proxy."
- Unsafe: "persistent homology prior," "cubical PH result," "persistence-based segmentation," unless genuine cubical PH is implemented.

If the paper aims to include Morse/cubical PH, that remains **not yet implemented** in the current Phase 2 evidence.

### Prior-Art Positioning

Verified source:

- Francois and Tinarrage, "Train-Free Segmentation in MRI with Cubical Persistent Homology," arXiv:2401.01160. The paper explicitly studies MRI segmentation with cubical PH, includes glioblastoma examples, uses a spherical ET topology model, and reports that their assumptions hold on 441/1251 images (35.3%). Source: https://arxiv.org/abs/2401.01160

Positioning:

- Their work uses actual cubical persistent homology and representative-cycle/topological reasoning.
- This project currently uses classical ET predictions plus learned normalized shape proxies to reduce false positives.
- This project's negative finding is useful: post-treatment ET and classical predictions often violate the spherical/clean-component assumption.

Other prior-art items requested:

- SEDT-3 tumor-shape PH: needs exact bibliographic verification before citation. The general SEDT + PH idea is real in topology-estimation literature, but I did not verify a tumor-shape paper matching the exact label "SEDT-3" in this audit.
- Prastawa atlas: needs exact bibliographic verification before citation. Do not cite from memory.
- SRI24 ET-occurrence atlas built from 1251 BraTS21 cases: the 1251-case figure and SRI template context appear in the train-free PH source above, but a separate atlas paper/source should be verified before claiming an independent SRI24 ET-occurrence atlas contribution.

Gap:

- The prior-art bibliography is not paper-ready. It needs a cleaned `.bib` pass with exact titles, venues, years, and claims.

### Assumptions Audit

| Assumption | Status in data | Consequence |
|---|---|---|
| ET is compact/spherical | Violated often in post-treatment and multifocal cases | Hard shape filter kills detection; PH/sphere claims must be scoped |
| Enhancement `T1c - T1n` localizes ET | Violated by non-tumor enhancement | 00533/02078 show brightest structure can be non-tumor |
| Population location prior transfers | Weak | P1 atlas was diffuse; seeds not in bottom atlas percentiles |
| Shape separates FP from true ET | Partly true | Component-level signal exists, but classical true-tumor predictions can be shape-corrupted |
| Global Dice summarizes performance | Violated | Absent and small ET make pooled Dice clinically misleading |
| Classical methods can "call absent" | Violated | 0/33 absent cases were called empty by any baseline method |

## D. Reproducibility Claim Verification

### What Is Verified

- Regression fixture tests exist for three pinned cases and guard branch checks where recorded.
- Determinism test checks subprocess bit-identical masks for a fixture case.
- Runner determinism test invokes `run_all.py` twice, but with `--skip-clean`.
- Stage 5 runner validation shows bit-identical masks to the frozen baseline for the original 20 core-method cases.
- 100-case provenance records command, data root, clean root, output root, git hash, config hash, preproc hash, case list, requirements lock, and timings.
- Cohort manifest is fully specified by IDs and measured strata; it is safe to version and sufficient to reconstruct the 100/150 split given access to the gated BraTS data.

### Overstated Or Not Yet Verified

- End-to-end raw-data determinism, including N4/cleaning, is **not proven** by the visible tests. The runner test uses `--skip-clean`; the 20-case runner provenance also uses `--skip-clean`.
- The 100-case run provenance includes a raw-data command and preproc hash, but it is not a bit-identical repeat proof from raw inputs.
- Provenance captures `requirements-lock`, but it does not guarantee OS-level/SimpleITK/ITK threading behavior is identical across machines.
- The repository is very dirty; a paper/release claim needs a clean tag/commit.

## E. Prioritized Fix List

1. **Lock P2b operating point before validation.** Freeze `method=otsu_T1c`, threshold `0.010`, metric, and cohort protocol, then evaluate on an untouched cohort or at least a new nested split. This is the biggest review risk.
2. **Add confidence intervals to all Stage 4 baseline tables.** Especially absent FP flood rates, median FP volumes, and per-stratum lesion-wise metrics.
3. **Run sensitivity excluding the 3 original-20 overlap cases.** Report whether Stage 4/P2 conclusions change without `02086/02143/02151`.
4. **Correct terminology.** Replace any "PH/persistence prior" wording for current P2 with "shape/topology-inspired proxies." Only use PH language after implementing cubical PH.
5. **Bibliography hardening.** Verify and cite exact prior-art records for SEDT-3, Prastawa atlas, and SRI24 ET atlas claims. Keep Francois/Tinarrage as the anchor PH comparison.
6. **End-to-end reproducibility proof.** Add a raw-input runner determinism test for at least one case that includes cleaning/N4 or explicitly limit the claim to cleaned-input determinism.
7. **Multiplicity disclosure.** Add a methods paragraph stating P1/P2 sweeps are exploratory and uncorrected; only locked validation should carry confirmatory claims.
8. **Avoid tiny-n headlines.** Move large-unifocal n=2 and peri-cavity n=4 to qualitative/mechanistic sections.
9. **Release manifest hygiene.** Keep `COHORT_MANIFEST.md` and `COHORT_MANIFEST_selected.csv`; do not commit gated data or generated raw masks unless allowed.

## Paper-Safe Claim Wording

Safe:

> In this stratified 100-case post-treatment BraTS-2024 GLI cohort, all five classical ET methods produced nonempty predictions on every ET-absent case, with absent-case flood rates of 93.9-100%. This supports false-positive burden, not pooled Dice, as the central failure mode.

Safe but suggestive:

> Shape-normalized component plausibility produced a more favorable exploratory FP/detection tradeoff than a spatial occurrence atlas, but the selected operating point was post hoc and residual absent-case flood remained 78.8%.

Unsafe:

> A persistent-homology prior solved false positives.

Unsafe:

> P2b is clinically usable or robustly validated.

Unsafe:

> Peri-cavity FP separation is proven by AUC 1.000.
