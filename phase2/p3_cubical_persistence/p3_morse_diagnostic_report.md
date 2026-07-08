# P3 Cubical Persistence / Morse Diagnostic

This is the first genuine topological computation in the project: GUDHI `CubicalComplex` is run on the normalized enhancement map (`max(T1c - T1n, 0)` scaled by its positive p99.5). For each connected component, PH is computed on the component crop with outside-component voxels set to zero. Superlevel-set H0 persistence is obtained by running sublevel PH on `-enhancement`; the essential H0 class is capped at background, giving a normalized peak-prominence score in `[0,1]`.

Prediction masks with many components use the diagnostic candidate policy: top 5 by volume union top 5 by enhancement. Components outside that candidate set are rejected by the count/rank prior. Large crops above the exact voxel budget are deterministically downsampled before PH; the downsample step is recorded per component.

## Diagnostic Verdict

Verdict: **persistence_overlaps_or_confound_high_wrap_up**.

| comparison        | score                 | n_true_components | n_confound_components | true_median | true_p25 | true_p75 | confound_median | confound_p25 | confound_p75 | auc_true_higher_than_confound | mannwhitney_p | diagnostic_verdict                            |
| ----------------- | --------------------- | ----------------- | --------------------- | ----------- | -------- | -------- | --------------- | ------------ | ------------ | ----------------------------- | ------------- | --------------------------------------------- |
| absent_fp         | ph_h0_max_persistence | 183               | 675                   | 0.5831      | 0.2042   | 0.8954   | 1.0000          | 1.0000       | 1.0000       | 0.1263                        | 0.0000        | persistence_overlaps_or_confound_high_wrap_up |
| peri_cavity_fp_n4 | ph_h0_max_persistence | 183               | 4                     | 0.5831      | 0.2042   | 0.8954   | 1.0000          | 1.0000       | 1.0000       | 0.0820                        | 0.0042        | persistence_overlaps_or_confound_high_wrap_up |
| all_confound      | ph_h0_max_persistence | 183               | 679                   | 0.5831      | 0.2042   | 0.8954   | 1.0000          | 1.0000       | 1.0000       | 0.1260                        | 0.0000        | persistence_overlaps_or_confound_high_wrap_up |

Interpretation rule: if confound components have lower H0 persistence than true ET (AUC >= 0.75 for true > confound), persistence supports pushing toward Morse. If confounds are high-persistence or overlap true ET, intensity persistence will inherit the confound and Morse should be treated as unlikely to solve the core FP problem.

## Train-Fold Count/Rank Prior

The raw persistence-gap heuristic was not a useful standalone count rule. The brightest scored component is usually maximally persistent (`top_score` median 1.0), and the gap either collapses to `k=1` or points late in the candidate list for flood-prone methods. Therefore the applied prior uses train-fold selection of `k` rather than choosing `k` on held-out cases.

| method             | n_case_methods | median_largest_gap_k | frac_gap_k_eq_1 | median_largest_gap | median_top_score | median_second_score |
| ------------------ | -------------- | -------------------- | --------------- | ------------------ | ---------------- | ------------------- |
| gmm_2d             | 100            | 8.0000               | 0.1400          | 0.1258             | 1.0000           | 1.0000              |
| gmm_T1c            | 100            | 1.0000               | 1.0000          |                    | 1.0000           |                     |
| otsu_T1c           | 100            | 1.0000               | 1.0000          |                    | 1.0000           |                     |
| sustraccion        | 99             | 8.0000               | 0.2020          | 0.0972             | 1.0000           | 1.0000              |
| variational_spline | 100            | 1.0000               | 1.0000          |                    | 1.0000           |                     |

For each held-out fold and method, `k` was selected on the other four folds only: minimize absent flood, then median FP, subject to preserving >=90% of that method's train-fold large-lesion baseline where feasible.

| holdout_fold | method             | selected_k_for_fold_method | train_absent_flood_rate | train_absent_median_fp_vox | train_large_retention_vs_own_baseline | selection_policy                                         |
| ------------ | ------------------ | -------------------------- | ----------------------- | -------------------------- | ------------------------------------- | -------------------------------------------------------- |
| 0            | otsu_T1c           | 1.0000                     | 0.9231                  | 14165.5000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 0            | gmm_T1c            | 1.0000                     | 0.9615                  | 19351.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 0            | sustraccion        | 1.0000                     | 0.9231                  | 63104.0000                 | 18.6904                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 0            | gmm_2d             | 1.0000                     | 0.9231                  | 71628.0000                 | 18.2891                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 0            | variational_spline | 1.0000                     | 0.9615                  | 19352.5000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 1            | otsu_T1c           | 1.0000                     | 0.9615                  | 16554.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 1            | gmm_T1c            | 1.0000                     | 1.0000                  | 19351.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 1            | sustraccion        | 1.0000                     | 0.9231                  | 66272.5000                 | 18.0037                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 1            | gmm_2d             | 1.0000                     | 0.8846                  | 79611.5000                 | 18.1542                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 1            | variational_spline | 1.0000                     | 1.0000                  | 19352.5000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 2            | otsu_T1c           | 1.0000                     | 0.9615                  | 14125.5000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 2            | gmm_T1c            | 1.0000                     | 0.9615                  | 19413.5000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 2            | sustraccion        | 1.0000                     | 0.8846                  | 70116.5000                 | 19.3841                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 2            | gmm_2d             | 1.0000                     | 0.9231                  | 76141.0000                 | 17.4066                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 2            | variational_spline | 1.0000                     | 0.9615                  | 19415.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 3            | otsu_T1c           | 1.0000                     | 0.9259                  | 13841.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 3            | gmm_T1c            | 1.0000                     | 0.9630                  | 18512.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 3            | sustraccion        | 1.0000                     | 0.8889                  | 60070.0000                 | 14.0357                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 3            | gmm_2d             | 1.0000                     | 0.9259                  | 72438.0000                 | 17.8105                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 3            | variational_spline | 1.0000                     | 0.9630                  | 18518.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 4            | otsu_T1c           | 1.0000                     | 0.9259                  | 13855.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 4            | gmm_T1c            | 1.0000                     | 0.9630                  | 18837.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 4            | sustraccion        | 1.0000                     | 0.9259                  | 61533.0000                 | 35.4782                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 4            | gmm_2d             | 1.0000                     | 0.8889                  | 70818.0000                 | 27.7922                               | min_flood_then_median_fp_subject_to_train_large_ge_90pct |
| 4            | variational_spline | 1.0000                     | 0.9630                  | 18837.0000                 | 1.0000                                | min_flood_then_median_fp_subject_to_train_large_ge_90pct |

## Held-Out Per-Axis Comparison

| method             | construction_type       | absent_flood_rate | absent_median_fp_vox | delta_absent_flood_vs_otsu | delta_absent_flood_ci_low | delta_absent_flood_ci_high | large_lesionwise_mean | large_retention_vs_detection_baseline | large_detection_preserved_90pct |
| ------------------ | ----------------------- | ----------------- | -------------------- | -------------------------- | ------------------------- | -------------------------- | --------------------- | ------------------------------------- | ------------------------------- |
| gmm_2d             | intensity/statistical   | 0.9091            | 72438.0000           | -0.0303                    | -0.1818                   | 0.0909                     | 0.0821                | 0.4657                                | False                           |
| gmm_T1c            | intensity/statistical   | 0.9697            | 18837.0000           | 0.0303                     | 0.0000                    | 0.0909                     | 0.1707                | 0.9684                                | True                            |
| otsu_T1c           | intensity/statistical   | 0.9394            | 13935.0000           | 0.0000                     | 0.0000                    | 0.0000                     | 0.1650                | 0.9364                                | True                            |
| sustraccion        | intensity/statistical   | 0.9091            | 64675.0000           | -0.0303                    | -0.1515                   | 0.0909                     | 0.1721                | 0.9763                                | True                            |
| variational_spline | deformable/region-based | 0.9697            | 18837.0000           | 0.0303                     | 0.0000                    | 0.0909                     | 0.1762                | 1.0000                                | True                            |

## P3 vs P2b

| prior                                | method             | absent_flood_rate | absent_median_fp_vox | large_lesionwise | large_detection_ratio | large_detection_preserved_90pct | selection                                     |
| ------------------------------------ | ------------------ | ----------------- | -------------------- | ---------------- | --------------------- | ------------------------------- | --------------------------------------------- |
| P2b soft shape                       | variational_spline | 0.7273            | 14118.0000           | 0.1632           | 0.9261                | True                            | global_best_fp_while_large_detection_ge_90pct |
| P3 cubical H0 persistence count/rank | sustraccion        | 0.9091            | 64675.0000           | 0.1721           | 0.9763                | True                            | best_fp_with_large_detection_ge_90pct         |

## Method-Type Interaction

| construction_type       | n_methods | mean_absent_flood_rate | best_absent_flood_rate | mean_absent_median_fp_vox | mean_large_retention_vs_detection_baseline | n_methods_preserve_large_90pct | best_method_by_fp_with_90pct_gate |
| ----------------------- | --------- | ---------------------- | ---------------------- | ------------------------- | ------------------------------------------ | ------------------------------ | --------------------------------- |
| deformable/region-based | 1         | 0.9697                 | 0.9697                 | 18837.0000                | 1.0000                                     | 1                              | variational_spline                |
| intensity/statistical   | 4         | 0.9318                 | 0.9091                 | 42471.2500                | 0.8367                                     | 3                              | sustraccion                       |
