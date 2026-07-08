# Phase 2 Final Data Dictionary

This directory is the canonical data layer for the paper reports. Files here are copied from locked Stage-4 and Phase-2 artifacts after consolidation. Intermediate and superseded Phase-2 CSVs were moved to `phase2/_archive/` and indexed in `phase2/_archive/ARCHIVE_INDEX.csv`.

## Canonical Files

### `baseline_targets_per_axis.csv`

- Source: `phase2\baseline_targets_per_axis.csv`
- Rows: 20
- Purpose: Per-axis baseline targets used for Phase 2 comparisons.
- Units: rates and Dice unitless; FP volume in voxels

| column | description | units |
| --- | --- | --- |
| `axis` | Axis. | unitless/text |
| `target_scope` | Target scope. | unitless/text |
| `metodo` | Segmentation method name. | categorical |
| `is_axis_best` | Boolean flag for the stated condition. | boolean |
| `axis_primary_metric` | Axis primary metric. | unitless/text |
| `axis_primary_direction` | Axis primary direction. | unitless/text |
| `n` | Count of cases, lesions, components, or events. | count |
| `lesionwise_dice_mean` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `lesionwise_dice_median` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `global_dice_mean` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `global_dice_median` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `hd95_median_defined` | Surface/distance metric. | millimeters |
| `lesion_tp` | Count of cases, lesions, components, or events. | count |
| `lesion_fn` | Count of cases, lesions, components, or events. | count |
| `lesion_fp` | Count of cases, lesions, components, or events. | count |
| `absent_flood_gt_10000_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `absent_fp_volume_median` | False-positive predicted ET volume. | voxels |
| `absent_fp_volume_max` | False-positive predicted ET volume. | voxels |

### `cohort_manifest_selected.csv`

- Source: `cohort\COHORT_MANIFEST_selected.csv`
- Rows: 150
- Purpose: Selected 100-case process cohort plus 50 reserve cases/fold metadata.
- Units: case-level cohort metadata; voxels/mm are SRI24 1 mm grid units

| column | description | units |
| --- | --- | --- |
| `case_id` | BraTS case identifier. | identifier |
| `process` | Process. | unitless/text |
| `fold` | Cross-validation/process fold assignment. | fold index |
| `stratum` | Stratum. | unitless/text |
| `vol_bin` | ET-volume stratum: absent, small, medium, or large. | categorical |
| `focality` | GT focality class from ET connected components. | categorical |
| `confound_bin` | Confound bin. | unitless/text |
| `et_present` | Et present. | unitless/text |
| `et_mm3` | Surface/distance metric. | millimeters |
| `et_num_components` | Count of cases, lesions, components, or events. | count |
| `rc_present` | Rc present. | unitless/text |
| `et_to_rc_min_mm` | Surface/distance metric. | millimeters |
| `enh_confound_risk` | Enh confound risk. | unitless/text |

### `four_regime_discrimination_diagnostic.csv`

- Source: `phase2\four_regime\four_regime_discrimination_diagnostic.csv`
- Rows: 4
- Purpose: Regime-level separability/impossibility-chain diagnostic.
- Units: AUC/rank/percentile values as text with n and robustness labels

| column | description | units |
| --- | --- | --- |
| `regime_id` | Regime id. | unitless/text |
| `regime_label` | Human-readable label, verdict, interpretation, or category. | text/categorical |
| `discrimination_question` | Human-readable label, verdict, interpretation, or category. | text/categorical |
| `separability_verdict` | Human-readable label, verdict, interpretation, or category. | text/categorical |
| `separability_metric` | Separability metric. | unitless/text |
| `separability_value` | Separability value. | unitless/text |
| `n` | Count of cases, lesions, components, or events. | count |
| `robustness` | Audit robustness class or qualitative support label. | categorical/text |
| `interpretation` | Human-readable label, verdict, interpretation, or category. | text/categorical |

### `four_regime_master_comparison.csv`

- Source: `phase2\four_regime\four_regime_master_comparison.csv`
- Rows: 5
- Purpose: Method-level comparison across R1 intensity, R2 spatial, R3 shape-proxy, R4 cubical persistence.
- Units: rates and retention unitless; FP volume in voxels

| column | description | units |
| --- | --- | --- |
| `method` | Segmentation method name. | categorical |
| `R1_regime_label` | Human-readable label, verdict, interpretation, or category. | text/categorical |
| `R1_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `R1_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `R1_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `R1_large_detection_retention_vs_R1_best` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `R1_operating_point` | R1 operating point. | unitless/text |
| `R1_valid_operating_point` | Boolean flag for the stated condition. | boolean |
| `R2_regime_label` | Human-readable label, verdict, interpretation, or category. | text/categorical |
| `R2_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `R2_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `R2_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `R2_large_detection_retention_vs_R1_best` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `R2_operating_point` | R2 operating point. | unitless/text |
| `R2_valid_operating_point` | Boolean flag for the stated condition. | boolean |
| `R3_regime_label` | Human-readable label, verdict, interpretation, or category. | text/categorical |
| `R3_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `R3_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `R3_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `R3_large_detection_retention_vs_R1_best` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `R3_operating_point` | R3 operating point. | unitless/text |
| `R3_valid_operating_point` | Boolean flag for the stated condition. | boolean |
| `R4_regime_label` | Human-readable label, verdict, interpretation, or category. | text/categorical |
| `R4_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `R4_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `R4_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `R4_large_detection_retention_vs_R1_best` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `R4_operating_point` | R4 operating point. | unitless/text |
| `R4_valid_operating_point` | Boolean flag for the stated condition. | boolean |

### `four_regime_summary.csv`

- Source: `phase2\four_regime\four_regime_summary.csv`
- Rows: 4
- Purpose: Regime-level best achievable FP and detection summaries.
- Units: rates and retention unitless; FP volume in voxels

| column | description | units |
| --- | --- | --- |
| `regime_id` | Regime id. | unitless/text |
| `regime_label` | Human-readable label, verdict, interpretation, or category. | text/categorical |
| `precise_methodological_label` | Human-readable label, verdict, interpretation, or category. | text/categorical |
| `source` | Source path or provenance field. | path/text |
| `best_fp_method` | Statistical test p-value. | unitless |
| `best_fp_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `best_fp_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `best_fp_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `best_fp_retention_vs_R1_best_detection` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `fp_flood_beats_R1_best` | Statistical test p-value. | unitless |
| `fp_median_beats_R1_best` | Statistical test p-value. | unitless |
| `fp_axis_true_beat_R1` | Statistical test p-value. | unitless |
| `fp_axis_robustness_label` | Statistical test p-value. | unitless |
| `best_detection_method` | Best detection method. | unitless/text |
| `best_detection_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `best_detection_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `best_detection_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `detection_beats_R1_best` | Detection beats r1 best. | unitless/text |
| `detection_ties_R1_best` | Detection ties r1 best. | unitless/text |
| `detection_axis_robustness_label` | Audit robustness class or qualitative support label. | categorical/text |

### `growth_metric_improvement_from_baseline.csv`

- Source: `phase2\growth_metric\growth_metric_improvement_from_baseline.csv`
- Rows: 5
- Purpose: Best prior-assisted result per method and axis relative to baseline.
- Units: deltas in same units as source axis

| column | description | units |
| --- | --- | --- |
| `method` | Segmentation method name. | categorical |
| `baseline_flood` | Baseline flood. | unitless/text |
| `best_prior_for_fp` | Best prior for fp. | unitless/text |
| `best_prior_flood` | Best prior flood. | unitless/text |
| `delta_flood` | Difference relative to a baseline comparator. | same as metric axis |
| `baseline_median_fp` | Summary statistic over the relevant cohort subset. | see metric name |
| `best_prior_median_fp` | Summary statistic over the relevant cohort subset. | see metric name |
| `delta_median_fp` | Summary statistic over the relevant cohort subset. | see metric name |
| `best_fp_large_lesionwise` | Statistical test p-value. | unitless |
| `best_fp_preserves_90pct_of_best_detector` | Statistical test p-value. | unitless |
| `baseline_large_lesionwise` | Baseline large lesionwise. | unitless/text |
| `best_prior_for_detection` | Best prior for detection. | unitless/text |
| `best_prior_large_lesionwise` | Best prior large lesionwise. | unitless/text |
| `delta_large_lesionwise` | Difference relative to a baseline comparator. | same as metric axis |
| `best_detection_crosses_90pct_of_best_detector` | Best detection crosses 90pct of best detector. | unitless/text |
| `clean_usable_window` | Clean usable window. | unitless/text |
| `loose_usable_window` | Loose usable window. | unitless/text |
| `fp_improvement_label` | Statistical test p-value. | unitless |
| `detection_improvement_label` | Human-readable label, verdict, interpretation, or category. | text/categorical |

### `growth_metric_table.csv`

- Source: `phase2\growth_metric\growth_metric_table.csv`
- Rows: 5
- Purpose: Per-method trajectories across baseline, P1, P2b, and P3.
- Units: rates and Dice unitless; FP volume in voxels

| column | description | units |
| --- | --- | --- |
| `method` | Segmentation method name. | categorical |
| `baseline_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `baseline_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `baseline_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `baseline_operating_point` | Baseline operating point. | unitless/text |
| `baseline_valid_operating_point` | Boolean flag for the stated condition. | boolean |
| `p1_spatial_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `p1_spatial_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `p1_spatial_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `p1_spatial_operating_point` | P1 spatial operating point. | unitless/text |
| `p1_spatial_valid_operating_point` | Boolean flag for the stated condition. | boolean |
| `p2b_soft_shape_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `p2b_soft_shape_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `p2b_soft_shape_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `p2b_soft_shape_operating_point` | P2b soft shape operating point. | unitless/text |
| `p2b_soft_shape_valid_operating_point` | Boolean flag for the stated condition. | boolean |
| `p3_cubical_persistence_absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `p3_cubical_persistence_absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `p3_cubical_persistence_large_lesionwise_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `p3_cubical_persistence_operating_point` | P3 cubical persistence operating point. | unitless/text |
| `p3_cubical_persistence_valid_operating_point` | Boolean flag for the stated condition. | boolean |

### `p1_fixed_detection_cost_operating_points.csv`

- Source: `phase2\p1_spatial_atlas\tradeoff_analysis\p1_fixed_detection_cost_operating_points.csv`
- Rows: 3
- Purpose: P1 spatial-prior operating points at fixed detection cost.
- Units: rates unitless; FP volume in voxels

| column | description | units |
| --- | --- | --- |
| `mode` | Mode. | unitless/text |
| `method` | Segmentation method name. | categorical |
| `selection` | Selection. | unitless/text |
| `chosen_threshold` | Operating threshold used by the method/prior. | method-specific |
| `large_baseline` | Large baseline. | unitless/text |
| `large_90pct_target` | Large 90pct target. | unitless/text |
| `large_lesionwise` | Large lesionwise. | unitless/text |
| `large_detection_ratio` | Large detection ratio. | unitless/text |
| `absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `present_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `large_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `first_threshold_large_drop_gt_10pct` | Statistical test p-value. | unitless |
| `flood_at_first_large_drop_gt_10pct` | Statistical test p-value. | unitless |

### `p2_shape_separation_tests.csv`

- Source: `phase2\p2_shape_probe\p2_shape_separation_tests.csv`
- Rows: 14
- Purpose: Pre-P2 component shape separability tests.
- Units: AUC unitless; p-values unitless; n is component count

| column | description | units |
| --- | --- | --- |
| `comparison` | Comparison. | unitless/text |
| `metric` | Metric. | unitless/text |
| `true_median` | Summary statistic over the relevant cohort subset. | see metric name |
| `false_median` | Summary statistic over the relevant cohort subset. | see metric name |
| `true_q25` | True q25. | unitless/text |
| `true_q75` | True q75. | unitless/text |
| `false_q25` | False q25. | unitless/text |
| `false_q75` | False q75. | unitless/text |
| `mann_whitney_alt_true_greater_p` | Statistical test p-value. | unitless |
| `auc_probability_true_greater` | Area under the ROC curve; true ET ranked above confound when specified. | unitless [0,1] |
| `n_true` | Count of cases, lesions, components, or events. | count |
| `n_false` | Count of cases, lesions, components, or events. | count |

### `p2_soft_shape_operating_points.csv`

- Source: `phase2\p2_soft_shape_sweep\p2_soft_shape_operating_points.csv`
- Rows: 6
- Purpose: P2b soft shape-proxy operating points.
- Units: rates and retention unitless; FP volume in voxels

| column | description | units |
| --- | --- | --- |
| `method` | Segmentation method name. | categorical |
| `shape_score_threshold` | Operating threshold used by the method/prior. | method-specific |
| `n_cases` | Count of cases, lesions, components, or events. | count |
| `absent_n` | Absent n. | unitless/text |
| `absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `absent_max_fp_vox` | False-positive predicted ET volume. | voxels |
| `present_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `small_n` | Small n. | unitless/text |
| `small_lesionwise_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `small_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `small_lesion_tp` | Small lesion tp. | unitless/text |
| `small_lesion_fn` | Small lesion fn. | unitless/text |
| `small_lesion_fp` | Small lesion fp. | unitless/text |
| `medium_n` | Medium n. | unitless/text |
| `medium_lesionwise_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `medium_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `medium_lesion_tp` | Medium lesion tp. | unitless/text |
| `medium_lesion_fn` | Medium lesion fn. | unitless/text |
| `medium_lesion_fp` | Medium lesion fp. | unitless/text |
| `large_n` | Large n. | unitless/text |
| `large_lesionwise_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `large_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `large_lesion_tp` | Large lesion tp. | unitless/text |
| `large_lesion_fn` | Large lesion fn. | unitless/text |
| `large_lesion_fp` | Large lesion fp. | unitless/text |
| `large_detection_ratio` | Large detection ratio. | unitless/text |
| `keeps_large_detection_90pct` | Keeps large detection 90pct. | unitless/text |
| `selection` | Selection. | unitless/text |
| `delta_absent_flood_vs_otsu` | Difference relative to a baseline comparator. | same as metric axis |
| `delta_absent_flood_ci_low` | Difference relative to a baseline comparator. | same as metric axis |
| `delta_absent_flood_ci_high` | Difference relative to a baseline comparator. | same as metric axis |
| `paired_flood_mcnemar_p` | Statistical test p-value. | unitless |
| `delta_absent_median_fp_vs_otsu` | Statistical test p-value. | unitless |
| `paired_absent_fp_vox_delta_median_ci_low` | False-positive predicted ET volume. | voxels |
| `paired_absent_fp_vox_delta_median_ci_high` | False-positive predicted ET volume. | voxels |
| `paired_fp_vox_wilcoxon_p` | False-positive predicted ET volume. | voxels |
| `small_detection_baseline_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `small_delta_vs_detection_best` | Difference relative to a baseline comparator. | same as metric axis |
| `small_delta_ci_low` | Difference relative to a baseline comparator. | same as metric axis |
| `small_delta_ci_high` | Difference relative to a baseline comparator. | same as metric axis |
| `small_paired_wilcoxon_p` | Statistical test p-value. | unitless |
| `small_baseline_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `small_dominant_found_delta` | Difference relative to a baseline comparator. | same as metric axis |
| `medium_detection_baseline_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `medium_delta_vs_detection_best` | Difference relative to a baseline comparator. | same as metric axis |
| `medium_delta_ci_low` | Difference relative to a baseline comparator. | same as metric axis |
| `medium_delta_ci_high` | Difference relative to a baseline comparator. | same as metric axis |
| `medium_paired_wilcoxon_p` | Statistical test p-value. | unitless |
| `medium_baseline_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `medium_dominant_found_delta` | Difference relative to a baseline comparator. | same as metric axis |
| `large_detection_baseline_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `large_delta_vs_detection_best` | Difference relative to a baseline comparator. | same as metric axis |
| `large_delta_ci_low` | Difference relative to a baseline comparator. | same as metric axis |
| `large_delta_ci_high` | Difference relative to a baseline comparator. | same as metric axis |
| `large_paired_wilcoxon_p` | Statistical test p-value. | unitless |
| `large_baseline_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `large_dominant_found_delta` | Difference relative to a baseline comparator. | same as metric axis |
| `clears_clean_bar` | Clears clean bar. | unitless/text |

### `p3_key_comparison_vs_baseline.csv`

- Source: `phase2\p3_cubical_persistence\p3_key_comparison_vs_baseline.csv`
- Rows: 5
- Purpose: P3 count/rank prior comparison against baseline.
- Units: rates and Dice unitless; FP volume in voxels

| column | description | units |
| --- | --- | --- |
| `method` | Segmentation method name. | categorical |
| `construction_type` | Construction type. | unitless/text |
| `n_cases` | Count of cases, lesions, components, or events. | count |
| `absent_n` | Absent n. | unitless/text |
| `absent_flood_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `absent_median_fp_vox` | False-positive predicted ET volume. | voxels |
| `absent_max_fp_vox` | False-positive predicted ET volume. | voxels |
| `present_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `small_n` | Small n. | unitless/text |
| `small_lesionwise_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `small_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `small_lesion_tp` | Small lesion tp. | unitless/text |
| `small_lesion_fn` | Small lesion fn. | unitless/text |
| `small_lesion_fp` | Small lesion fp. | unitless/text |
| `medium_n` | Medium n. | unitless/text |
| `medium_lesionwise_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `medium_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `medium_lesion_tp` | Medium lesion tp. | unitless/text |
| `medium_lesion_fn` | Medium lesion fn. | unitless/text |
| `medium_lesion_fp` | Medium lesion fp. | unitless/text |
| `large_n` | Large n. | unitless/text |
| `large_lesionwise_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `large_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `large_lesion_tp` | Large lesion tp. | unitless/text |
| `large_lesion_fn` | Large lesion fn. | unitless/text |
| `large_lesion_fp` | Large lesion fp. | unitless/text |
| `delta_absent_flood_vs_otsu` | Difference relative to a baseline comparator. | same as metric axis |
| `delta_absent_flood_ci_low` | Difference relative to a baseline comparator. | same as metric axis |
| `delta_absent_flood_ci_high` | Difference relative to a baseline comparator. | same as metric axis |
| `delta_absent_median_fp_vs_otsu` | Statistical test p-value. | unitless |
| `paired_absent_fp_vox_delta_median_ci_low` | False-positive predicted ET volume. | voxels |
| `paired_absent_fp_vox_delta_median_ci_high` | False-positive predicted ET volume. | voxels |
| `paired_fp_vox_wilcoxon_p` | False-positive predicted ET volume. | voxels |
| `small_detection_baseline_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `small_retention_vs_detection_baseline` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `small_delta_vs_detection_best` | Difference relative to a baseline comparator. | same as metric axis |
| `small_delta_ci_low` | Difference relative to a baseline comparator. | same as metric axis |
| `small_delta_ci_high` | Difference relative to a baseline comparator. | same as metric axis |
| `small_baseline_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `small_dominant_found_delta` | Difference relative to a baseline comparator. | same as metric axis |
| `medium_detection_baseline_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `medium_retention_vs_detection_baseline` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `medium_delta_vs_detection_best` | Difference relative to a baseline comparator. | same as metric axis |
| `medium_delta_ci_low` | Difference relative to a baseline comparator. | same as metric axis |
| `medium_delta_ci_high` | Difference relative to a baseline comparator. | same as metric axis |
| `medium_baseline_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `medium_dominant_found_delta` | Difference relative to a baseline comparator. | same as metric axis |
| `large_detection_baseline_mean` | Summary statistic over the relevant cohort subset. | see metric name |
| `large_retention_vs_detection_baseline` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `large_delta_vs_detection_best` | Difference relative to a baseline comparator. | same as metric axis |
| `large_delta_ci_low` | Difference relative to a baseline comparator. | same as metric axis |
| `large_delta_ci_high` | Difference relative to a baseline comparator. | same as metric axis |
| `large_baseline_dominant_found_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `large_dominant_found_delta` | Difference relative to a baseline comparator. | same as metric axis |
| `large_detection_preserved_90pct` | Large detection preserved 90pct. | unitless/text |

### `p3_persistence_diagnostic_summary.csv`

- Source: `phase2\p3_cubical_persistence\p3_persistence_diagnostic_summary.csv`
- Rows: 3
- Purpose: P3 cubical H0 persistence separability diagnostic.
- Units: persistence scores normalized [0,1]; AUC/p-values unitless

| column | description | units |
| --- | --- | --- |
| `comparison` | Comparison. | unitless/text |
| `score` | Score. | unitless/text |
| `n_true_components` | Count of cases, lesions, components, or events. | count |
| `n_confound_components` | Count of cases, lesions, components, or events. | count |
| `true_median` | Summary statistic over the relevant cohort subset. | see metric name |
| `true_p25` | True p25. | unitless/text |
| `true_p75` | Summary statistic over the relevant cohort subset. | see metric name |
| `confound_median` | Summary statistic over the relevant cohort subset. | see metric name |
| `confound_p25` | Confound p25. | unitless/text |
| `confound_p75` | Summary statistic over the relevant cohort subset. | see metric name |
| `auc_true_higher_than_confound` | Area under the ROC curve; true ET ranked above confound when specified. | unitless [0,1] |
| `mannwhitney_p` | Statistical test p-value. | unitless |
| `diagnostic_verdict` | Human-readable label, verdict, interpretation, or category. | text/categorical |

### `stage4_absent_fp_summary.csv`

- Source: `analysis\stage4_metrics\stage4_absent_fp_summary.csv`
- Rows: 5
- Purpose: Stage-4 false-positive summary for ET-absent cases.
- Units: rates unitless; FP volumes in voxels

| column | description | units |
| --- | --- | --- |
| `metodo` | Segmentation method name. | categorical |
| `n_absent` | Count of cases, lesions, components, or events. | count |
| `correct_absent_pred_lt_10_vox` | Voxel count on the 1 mm grid. | voxels |
| `correct_absent_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `nonempty_pred_count` | Count of cases, lesions, components, or events. | count |
| `fp_volume_median` | False-positive predicted ET volume. | voxels |
| `fp_volume_p75` | False-positive predicted ET volume. | voxels |
| `fp_volume_p95` | False-positive predicted ET volume. | voxels |
| `fp_volume_max` | False-positive predicted ET volume. | voxels |
| `large_fp_gt_1000_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `flood_gt_10000_rate` | Fraction/rate over the relevant cases/components. | unitless [0,1] |

### `stage4_case_metrics.csv`

- Source: `analysis\stage4_metrics\stage4_case_metrics.csv`
- Rows: 500
- Purpose: Locked Stage-4 per-case, per-method metric table.
- Units: Dice/Jaccard unitless; volumes in voxels; HD95 in mm

| column | description | units |
| --- | --- | --- |
| `case_id` | BraTS case identifier. | identifier |
| `metodo` | Segmentation method name. | categorical |
| `vol_bin` | ET-volume stratum: absent, small, medium, or large. | categorical |
| `focality` | GT focality class from ET connected components. | categorical |
| `fold` | Cross-validation/process fold assignment. | fold index |
| `et_present_manifest` | Et present manifest. | unitless/text |
| `et_mm3_manifest` | Surface/distance metric. | millimeters |
| `gt_vox` | Voxel count on the 1 mm grid. | voxels |
| `pred_vox` | Voxel count on the 1 mm grid. | voxels |
| `global_dice` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `global_jaccard` | Jaccard overlap score; higher is better. | unitless [0,1] |
| `overseg_ratio` | Overseg ratio. | unitless/text |
| `correct_absent_pred_lt_10_vox` | Voxel count on the 1 mm grid. | voxels |
| `flood_gt_10000_vox` | Voxel count on the 1 mm grid. | voxels |
| `gt_components` | Count of cases, lesions, components, or events. | count |
| `pred_components` | Count of cases, lesions, components, or events. | count |
| `matched_components` | Count of cases, lesions, components, or events. | count |
| `lesion_tp` | Count of cases, lesions, components, or events. | count |
| `lesion_fn` | Count of cases, lesions, components, or events. | count |
| `lesion_fp` | Count of cases, lesions, components, or events. | count |
| `lesion_dice_sum` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `lesionwise_dice_mean` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `lesionwise_dice_median` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `lesion_detection_rate_dice_gt_0` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `lesion_detection_rate_dice_ge_0_1` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `hd95_mm` | Surface/distance metric. | millimeters |

### `stage4_present_by_vol_bin.csv`

- Source: `analysis\stage4_metrics\stage4_present_by_vol_bin.csv`
- Rows: 15
- Purpose: Stage-4 summary for ET-present strata.
- Units: means/medians over cases; HD95 in mm; ratios unitless

| column | description | units |
| --- | --- | --- |
| `metodo` | Segmentation method name. | categorical |
| `vol_bin` | ET-volume stratum: absent, small, medium, or large. | categorical |
| `n` | Count of cases, lesions, components, or events. | count |
| `global_dice_mean` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `global_dice_median` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `lesionwise_dice_mean` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `lesionwise_dice_median` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `lesion_detection_rate_mean` | Fraction/rate over the relevant cases/components. | unitless [0,1] |
| `hd95_median_defined` | Surface/distance metric. | millimeters |
| `hd95_undefined_count` | Surface/distance metric. | millimeters |
| `overseg_ratio_median` | Summary statistic over the relevant cohort subset. | see metric name |
| `n_global_dice_gt_0_4` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `n_lesionwise_dice_gt_0_4` | Dice-family overlap score; higher is better. | unitless [0,1] |

### `surface_dice_relationship_by_stratum.csv`

- Source: `phase2\surface_reconstruction\surface_dice_relationship_by_stratum.csv`
- Rows: 3
- Purpose: Spearman relationship between lesion-wise Dice and surface distances.
- Units: Spearman rho and p-values unitless

| column | description | units |
| --- | --- | --- |
| `vol_bin` | ET-volume stratum: absent, small, medium, or large. | categorical |
| `n_methods` | Count of cases, lesions, components, or events. | count |
| `spearman_lesionwise_vs_surface_asd_median` | Surface/distance metric. | millimeters |
| `p_asd` | Surface/distance metric. | millimeters |
| `spearman_lesionwise_vs_surface_hd95_median` | Surface/distance metric. | millimeters |
| `p_hd95` | Surface/distance metric. | millimeters |

### `surface_gt_reconstruction_floor_by_stratum.csv`

- Source: `phase2\surface_reconstruction\surface_gt_reconstruction_floor_by_stratum.csv`
- Rows: 3
- Purpose: Poisson reconstruction error floor from GT ET masks.
- Units: HD95/ASD in mm; Chamfer in squared mm distance convention

| column | description | units |
| --- | --- | --- |
| `vol_bin` | ET-volume stratum: absent, small, medium, or large. | categorical |
| `n` | Count of cases, lesions, components, or events. | count |
| `floor_hd95_mean` | Surface/distance metric. | millimeters |
| `floor_hd95_median` | Surface/distance metric. | millimeters |
| `floor_asd_mean` | Surface/distance metric. | millimeters |
| `floor_asd_median` | Surface/distance metric. | millimeters |
| `floor_chamfer_median` | Symmetric point-set Chamfer distance from reconstructed surfaces. | squared mm convention |
| `fallback_count` | Count of cases, lesions, components, or events. | count |

### `surface_prediction_fidelity_by_method_stratum.csv`

- Source: `phase2\surface_reconstruction\surface_prediction_fidelity_by_method_stratum.csv`
- Rows: 15
- Purpose: Prediction surface fidelity versus reconstructed GT by method and stratum.
- Units: HD95/ASD in mm; Chamfer in squared mm distance convention

| column | description | units |
| --- | --- | --- |
| `method` | Segmentation method name. | categorical |
| `vol_bin` | ET-volume stratum: absent, small, medium, or large. | categorical |
| `n` | Count of cases, lesions, components, or events. | count |
| `surface_hd95_mean` | Surface/distance metric. | millimeters |
| `surface_hd95_median` | Surface/distance metric. | millimeters |
| `surface_asd_mean` | Surface/distance metric. | millimeters |
| `surface_asd_median` | Surface/distance metric. | millimeters |
| `surface_chamfer_median` | Symmetric point-set Chamfer distance from reconstructed surfaces. | squared mm convention |
| `fallback_count` | Count of cases, lesions, components, or events. | count |
| `lesionwise_dice_mean` | Dice-family overlap score; higher is better. | unitless [0,1] |
| `global_dice_mean` | Dice-family overlap score; higher is better. | unitless [0,1] |

### `surface_prediction_fidelity_cases.csv`

- Source: `phase2\_archive\surface_reconstruction\surface_prediction_fidelity_cases.csv`
- Rows: 335
- Purpose: Case-level prediction surface fidelity versus reconstructed GT.
- Units: HD95/ASD in mm; Chamfer in squared mm distance convention

| column | description | units |
| --- | --- | --- |
| `case_id` | BraTS case identifier. | identifier |
| `vol_bin` | ET-volume stratum: absent, small, medium, or large. | categorical |
| `focality` | GT focality class from ET connected components. | categorical |
| `method` | Segmentation method name. | categorical |
| `pred_recon_points` | Pred recon points. | unitless/text |
| `gt_recon_points` | Gt recon points. | unitless/text |
| `pred_recon_status` | Pred recon status. | unitless/text |
| `pred_recon_fallback_used` | Pred recon fallback used. | unitless/text |
| `surface_hd95` | Surface/distance metric. | millimeters |
| `surface_asd` | Surface/distance metric. | millimeters |
| `surface_chamfer` | Symmetric point-set Chamfer distance from reconstructed surfaces. | squared mm convention |

## Archive Policy

`phase2/_archive/` stores intermediate CSVs that are not the canonical paper tables: per-case sweep dumps, component-decision tables, threshold-search outputs, and superseded summaries. They were moved, not deleted, to keep the paper-facing layer small while preserving the audit trail.