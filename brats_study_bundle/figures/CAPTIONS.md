# Figure Captions

## 01_impossibility_chain_four_regimes.png

Impossibility chain across the four methodological regimes. Intensity and location provide no safe separator for the hard confounds, shape proxies separate components offline but do not deploy cleanly, and genuine cubical H0 persistence is inverted: confounds are more persistent than true ET (AUC 0.126).

## 02_growth_metric_method_trajectories.png

Per-method trajectories across baseline, spatial prior, soft shape proxy, and cubical persistence on the locked Stage-4 axes. Curves are mostly flat or degenerate: false-positive reductions often cost detection, and detection does not materially improve.

## 03_best_vs_worst_variational_spline_vs_gmm2d.png

Best detector (variational_spline) versus worst baseline method (gmm_2d), with bootstrap 95% intervals from existing case-level results. The best method has higher large-lesion Dice and lower surface ASD, while gmm_2d carries much larger absent-case FP burden.

## 04_per_stratum_lesionwise_and_surface_asd.png

Per-stratum performance for the five core methods. Absent cases have no lesion-wise Dice or surface ASD because GT ET is absent; they must be scored by false-positive burden. Present-case detection improves mainly in the large-tumor stratum.

## 05_absent_case_flood_and_fp_volume.png

Absent-case hallucination result on 33 ET-absent cohort cases. Every method predicts nonempty ET, and 94-100% of absent cases exceed the 10,000-voxel flood threshold, although median FP volume differs by method.

## 06_poisson_surface_renders_good_vs_irreducible.png

Poisson surface render panel. Case 02306 shows a good large-tumor contrast case; 00533 and 02078 show irreducible non-tumor enhancement confounds reconstructing as coherent but spatially wrong surfaces.

## 07_surface_vs_dice_large_stratum.png

Large-stratum lesion-wise Dice versus surface ASD. The negative Spearman relationship shows that the surface metric confirms rather than rescues the voxel metric: methods with low lesion-wise Dice also have worse surface distance.
