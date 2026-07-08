# BraTS ET Redundancy Audit
## Phase 0 - Reconnaissance
- Metrics CSV: `output\tablas\metricas_ET.csv` with 200 rows, 20 cases, 10 methods.
- Predicted masks expected and present: 200/200.
- Naming convention: `output/segmentacion/<case>/<case>-et_<method>.nii.gz`.
- Prediction label convention: binary masks; see `prediction_label_inventory.csv` (`unique_values` is `0 1` for persisted masks).
- GT location: `images/<case>/<case>-seg.nii.gz`; ET is extracted as label `3` in code.
- GT labels observed on disk: ['0', '1', '2', '3', '4']; see `gt_label_inventory.csv`.
- Recomputed-vs-CSV max Dice abs diff: 0.000050; max Jaccard abs diff: 0.000050; see `metrics_recomputed_check.csv`.
- Methods in CSV: bspline, fast_marching, gmm_2d, gmm_T1c, level_set, otsu_T1c, rango_doble, spline, sustraccion, variational_spline.
- Method in code but not present in this run: semilla.

## Phase 1 - Method Inventory
| method | source | signals | family | key_params | shared_dependencies |
| --- | --- | --- | --- | --- | --- |
| otsu_T1c | src/brats_pipeline/seg_et_pipeline.py:81 | cleaned T1c | intensity-threshold | threshold_multiotsu classes=3; choose brightest class; morphology erosion=1 dilation=1 keep_largest=True | cleaned T1c; _morfologia |
| gmm_T1c | src/brats_pipeline/seg_et_pipeline.py:96 | cleaned T1c | mixture-model | GaussianMixture n_components=3 random_state=42; choose highest-mean cluster; morphology erosion=1 dilation=1 keep_largest=True | also used as roi_et_auto primary seed |
| sustraccion | src/brats_pipeline/seg_et_pipeline.py:117 | raw T1c, raw T1n -> mapa_dif | enhancement-threshold | joint normalization p=99.5; gaussian sigma=0.5; threshold positive mapa_dif at auto_pct=90; morphology keep_largest=False | _mapa_diferencia |
| gmm_2d | src/brats_pipeline/seg_et_pipeline.py:132 | cleaned T1c + mapa_dif | mixture-model | GaussianMixture n_components=4 random_state=42 max_iter=200; reject dif_mean>0.40 or t1c_mean<0.8*median; score=0.4*T1c+0.6*dif | _mapa_diferencia; cleaned T1c |
| rango_doble | src/brats_pipeline/seg_et_pipeline.py:185 | raw T1c, raw T1n -> mapa_dif | enhancement-threshold | lower threshold=70th percentile of positive mapa_dif; upper threshold=0.55; remove components <30 vox | _mapa_diferencia |
| fast_marching | src/brats_pipeline/seg_et_pipeline.py:233 | raw T1c, raw T1n -> mapa_dif; automatic or manual seed | seed/front | auto seed score=T1c*mapa_dif with 0.05<mapa<0.55 and uniform_filter size=5; tiempo_umbral=35.0; sigma=0.8 | _mapa_diferencia; semilla_automatica |
| semilla | src/brats_pipeline/seg_et_pipeline.py:318 | raw T1c, raw T1n -> mapa_dif; manual seed | seed/front | sphere radius=25; local percentile=65; morphology erosion=1 dilation=2 keep_largest=True | requires semilla_zyx; did not run in current CSV |
| level_set | src/brats_pipeline/seg_spline_levelset.py:212 | cleaned T1c + raw T1c/T1n through shared roi_et_auto/mapa_dif | deformable-contour | GAC prop=0.8 curv=3.0 adv=1.5 iters=120; sigmoid gradient alpha=-0.05 beta=0.1; bbox margin=12 | shared roi_et_auto and _post with other deformables |
| variational_spline | src/brats_pipeline/seg_spline_levelset.py:268 | shared roi_et_auto/mapa_dif | deformable-contour | morphological_chan_vese num_iter=35 smoothing=3 lambda1=lambda2=1.0; bbox margin=12 | shared roi_et_auto and _post with other deformables |
| bspline | src/brats_pipeline/seg_spline_levelset.py:333 | shared roi_et_auto/mapa_dif | deformable-contour | Chan-Vese num_iter=35 smoothing=2; per-slice cubic periodic B-spline smooth=3.0; pct_realce=78 | shared roi_et_auto and _post with other deformables; starts from Chan-Vese-like mask |
| spline | src/brats_pipeline/seg_spline_levelset.py:378 | shared roi_et_auto/mapa_dif | deformable-contour | active_contour alpha=0.05 beta=2.0 w_line=2.0 w_edge=1.0 gamma=0.02 max_iter=25; bbox margin=8 | shared roi_et_auto and _post with other deformables |

A-priori redundancy hypotheses before looking at mask agreement:
- `level_set`, `variational_spline`, `bspline`, and `spline` should be correlated because they share `roi_et_auto`, `mapa_dif`, and `_post` safeguards.
- `gmm_T1c` should be close to deformables whenever the shared ROI falls back to or accepts the GMM seed.
- `sustraccion` and `rango_doble` should share enhancement-map blind spots, but not necessarily identical masks because one is a one-sided high percentile and the other is a bounded interval.
- `fast_marching` may be complementary if the auto seed lands well, but vulnerable to seed/front leakage or undergrowth.

## Phase 2 - Quantitative Redundancy

### Performance Summary
| metodo | mean_dice | median_dice | std_dice | min_dice | max_dice | mean_jaccard | mean_runtime_s | cases | cases_gt_0_75 | cases_lt_0_25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| variational_spline | 0.5067 | 0.5063 | 0.2648 | 0.0000 | 0.8794 | 0.3788 | 55.0000 | 20 | 5 | 3 |
| otsu_T1c | 0.4854 | 0.5920 | 0.2938 | 0.0000 | 0.8212 | 0.3660 | 55.0000 | 20 | 6 | 5 |
| bspline | 0.4810 | 0.4651 | 0.2441 | 0.0000 | 0.8195 | 0.3482 | 55.0000 | 20 | 3 | 3 |
| gmm_T1c | 0.4810 | 0.5130 | 0.3013 | 0.0000 | 0.8655 | 0.3645 | 55.0000 | 20 | 6 | 5 |
| level_set | 0.4555 | 0.4748 | 0.2358 | 0.0000 | 0.7808 | 0.3229 | 55.0000 | 20 | 2 | 4 |
| spline | 0.4500 | 0.4840 | 0.2450 | 0.0000 | 0.7952 | 0.3206 | 55.0000 | 20 | 2 | 4 |
| sustraccion | 0.4467 | 0.4518 | 0.2027 | 0.0476 | 0.7918 | 0.3084 | 55.0000 | 20 | 2 | 4 |
| gmm_2d | 0.3738 | 0.3674 | 0.1849 | 0.0745 | 0.8282 | 0.2468 | 55.0000 | 20 | 1 | 5 |
| rango_doble | 0.2382 | 0.2264 | 0.1246 | 0.0961 | 0.5888 | 0.1412 | 55.0000 | 20 | 0 | 13 |
| fast_marching | 0.1846 | 0.1748 | 0.1283 | 0.0000 | 0.4855 | 0.1072 | 55.0000 | 20 | 0 | 15 |

### Top Prediction-Agreement Pairs
- bspline vs variational_spline: 0.9238
- gmm_T1c vs otsu_T1c: 0.8500
- level_set vs variational_spline: 0.8201
- bspline vs level_set: 0.8048
- gmm_T1c vs level_set: 0.7575
- level_set vs otsu_T1c: 0.7404
- gmm_T1c vs variational_spline: 0.7207
- bspline vs gmm_T1c: 0.7005
- otsu_T1c vs variational_spline: 0.6913
- spline vs variational_spline: 0.6887

### Top Performance-Correlation Pairs
- bspline vs variational_spline: 0.9564
- gmm_T1c vs spline: 0.9518
- gmm_T1c vs otsu_T1c: 0.9067
- bspline vs level_set: 0.8886
- level_set vs variational_spline: 0.8766
- otsu_T1c vs spline: 0.8616
- bspline vs gmm_T1c: 0.8315
- level_set vs spline: 0.8194
- gmm_T1c vs level_set: 0.8179
- level_set vs otsu_T1c: 0.8119

### Top Double-Fault Pairs
- fast_marching vs rango_doble: 0.9500
- fast_marching vs gmm_2d: 0.8000
- gmm_2d vs rango_doble: 0.8000
- fast_marching vs sustraccion: 0.6500
- rango_doble vs sustraccion: 0.6500
- fast_marching vs spline: 0.6000
- gmm_2d vs sustraccion: 0.6000
- bspline vs fast_marching: 0.5500
- bspline vs rango_doble: 0.5500
- fast_marching vs level_set: 0.5500

### Oracle
| cases | methods | best_single_method | best_single_mean_dice | oracle_mean_dice | headroom_oracle_minus_best_single |
| --- | --- | --- | --- | --- | --- |
| 20 | 10 | variational_spline | 0.5067 | 0.6320 | 0.1253 |

### Leave-One-Out Unique Value
| method | oracle_without_method | delta_full_minus_without | sole_best_cases | sole_above_075_cases |
| --- | --- | --- | --- | --- |
| otsu_T1c | 0.6218 | 0.0103 | 3 | 2 |
| sustraccion | 0.6226 | 0.0094 | 4 | 0 |
| gmm_2d | 0.6232 | 0.0088 | 3 | 0 |
| variational_spline | 0.6247 | 0.0074 | 4 | 1 |
| gmm_T1c | 0.6249 | 0.0072 | 4 | 0 |
| bspline | 0.6320 | 0.0000 | 0 | 0 |
| fast_marching | 0.6320 | 0.0000 | 0 | 0 |
| level_set | 0.6320 | 0.0000 | 0 | 0 |
| spline | 0.6320 | 0.0000 | 0 | 0 |
| rango_doble | 0.6320 | 0.0000 | 0 | 0 |

### Cluster Tree
- step 1: bspline + variational_spline at distance 0.0762 (n=2)
- step 2: gmm_T1c + otsu_T1c at distance 0.1500 (n=2)
- step 3: level_set + (bspline + variational_spline) at distance 0.1876 (n=3)
- step 4: (gmm_T1c + otsu_T1c) + (level_set + (bspline + variational_spline)) at distance 0.2880 (n=5)
- step 5: spline + ((gmm_T1c + otsu_T1c) + (level_set + (bspline + variational_spline))) at distance 0.3244 (n=6)
- step 6: gmm_2d + sustraccion at distance 0.4308 (n=2)
- step 7: rango_doble + (gmm_2d + sustraccion) at distance 0.5351 (n=3)
- step 8: (spline + ((gmm_T1c + otsu_T1c) + (level_set + (bspline + variational_spline)))) + (rango_doble + (gmm_2d + sustraccion)) at distance 0.7134 (n=9)
- step 9: fast_marching + ((spline + ((gmm_T1c + otsu_T1c) + (level_set + (bspline + variational_spline)))) + (rango_doble + (gmm_2d + sustraccion))) at distance 0.8542 (n=10)

### Surprises / Integrity Flags
- Found 49 exactly identical per-case mask pairs; see `identical_masks_by_case.csv`.
- Very high mean prediction agreement (>0.90): bspline/variational_spline=0.924.
- Runtime is identical for every method in the CSV. Code assigns one case-level elapsed time to all method rows, so `tiempo_s` is not per-method runtime (`run_all.py:168`).

## Artifacts
- `baseline`
- `double_fault.csv`
- `double_fault_heatmap.png`
- `gt_label_inventory.csv`
- `identical_masks_by_case.csv`
- `inventory_masks.csv`
- `metadata.json`
- `method_inventory.csv`
- `metrics_recomputed_check.csv`
- `oracle_by_case.csv`
- `oracle_leave_one_out.csv`
- `oracle_summary.csv`
- `perf_summary.csv`
- `performance_spearman.csv`
- `performance_spearman_heatmap.png`
- `prediction_agreement.csv`
- `prediction_agreement_cluster_tree.csv`
- `prediction_agreement_dendrogram.png`
- `prediction_agreement_heatmap.png`
- `prediction_label_inventory.csv`
- `report.md`
- `sole_above_075_cases.csv`
- `sole_best_cases.csv`
