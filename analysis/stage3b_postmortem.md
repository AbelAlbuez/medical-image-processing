# Stage 3B post-mortem and narrow salvage

Diagnostic replay only. No masks or baselines were regenerated.

## Step 1: structural read

The rejected 3B proxy was compared against the true best raw Chan-Vese iterate by Dice-vs-GT. For the three broken sacred cases, the proxy picked an earlier and smaller iterate than the Dice-optimal contour.

| case_id | group | proxy_iter | proxy_raw_voxels | proxy_raw_dice | best_raw_dice_iter | best_raw_voxels | best_raw_dice | proxy_minus_best_raw_iter | proxy_minus_best_raw_voxels | proxy_raw_dice_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BraTS-GLI-02108-100 | broken_sacred | 2 | 54546 | 0.6699 | 35 | 91701 | 0.8659 | -33 | -37155 | -0.1960 |
| BraTS-GLI-02116-100 | broken_sacred | 1 | 35468 | 0.5364 | 35 | 47611 | 0.7433 | -34 | -12143 | -0.2070 |
| BraTS-GLI-02151-100 | broken_sacred | 4 | 20707 | 0.6593 | 35 | 25940 | 0.7398 | -31 | -5233 | -0.0805 |
| BraTS-GLI-02118-100 | clean_large_comparator | 7 | 34637 | 0.8315 | 5 | 35257 | 0.8330 | 2 | -620 | -0.0015 |
| BraTS-GLI-02158-100 | clean_large_comparator | 5 | 50216 | 0.8230 | 7 | 49519 | 0.8249 | -2 | 697 | -0.0020 |

Conclusion: confirmed. The general best-iterate proxy is structurally biased against large late expansions. It badly harms the large-late sacred cases while offering only tiny selection error on the clean large comparators. The general Stage 3B approach is dead and should stay disabled.

## Step 2: 02169-only salvage

Current 02169 variational_spline baseline Dice is 0.6196. There are 02169 iterates that beat the current fallback and pass the 3A evidence metrics:

| iter | raw_voxels | raw_dice | evidence_lcc_fraction | evidence_enhancement_ratio | evidence_volume_multiple | post_voxels | post_dice | post_branch | post_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 21556 | 0.7455 | 0.9817 | 0.9955 | 0.3994 | 49134 | 0.6196 | collapse-detected | collapsed_small |
| 7 | 22830 | 0.7450 | 0.9640 | 1.0005 | 0.4253 | 21239 | 0.7686 | evolved | accepted |
| 8 | 21971 | 0.7450 | 0.9739 | 0.9998 | 0.4085 | 20643 | 0.7646 | evolved | accepted |
| 6 | 23314 | 0.7428 | 0.9542 | 1.0083 | 0.4365 | 21559 | 0.7688 | evolved | accepted |
| 10 | 20845 | 0.7423 | 0.9891 | 0.9947 | 0.3863 | 49134 | 0.6196 | collapse-detected | collapsed_small |
| 11 | 20501 | 0.7420 | 0.9961 | 0.9891 | 0.3794 | 49134 | 0.6196 | collapse-detected | collapsed_small |
| 12 | 19858 | 0.7365 | 0.9999 | 0.9906 | 0.3681 | 49134 | 0.6196 | collapse-detected | collapsed_small |
| 5 | 24540 | 0.7358 | 0.9304 | 1.0213 | 0.4621 | 22220 | 0.7702 | evolved | accepted |

This confirms 02169 is not signal-poor. However, using this information would require a selector, and the global selector is rejected. The only selector-free salvage is to relax the final-iterate collapse guard. The final iter-35 contour has raw Dice 0.6578 and evidence_volume_multiple about 0.295, so it is just below the current 0.40 collapsed_small cutoff.

Hypothetical collapse-threshold sensitivity, applied only in-memory:

| min_volume_frac | mean_dice | case_02169_dice | case_02169_delta | case_02169_voxels | case_02169_branch | case_02169_reason | n_evolved_cases_changed | n_evolved_cases_lost_dice | changed_non_evolved_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2000 | 0.5236 | 0.6582 | 0.0386 | 15529 | evolved | accepted | 0 | 0 | BraTS-GLI-02169-100 |
| 0.2500 | 0.5236 | 0.6582 | 0.0386 | 15529 | evolved | accepted | 0 | 0 | BraTS-GLI-02169-100 |
| 0.2800 | 0.5236 | 0.6582 | 0.0386 | 15529 | evolved | accepted | 0 | 0 | BraTS-GLI-02169-100 |
| 0.2900 | 0.5236 | 0.6582 | 0.0386 | 15529 | evolved | accepted | 0 | 0 | BraTS-GLI-02169-100 |
| 0.2950 | 0.5236 | 0.6582 | 0.0386 | 15529 | evolved | accepted | 0 | 0 | BraTS-GLI-02169-100 |
| 0.3000 | 0.5217 | 0.6196 | 0.0000 | 49134 | collapse-detected | collapsed_small | 0 | 0 |  |
| 0.3100 | 0.5217 | 0.6196 | 0.0000 | 49134 | collapse-detected | collapsed_small | 0 | 0 |  |
| 0.3500 | 0.5217 | 0.6196 | 0.0000 | 49134 | collapse-detected | collapsed_small | 0 | 0 |  |
| 0.4000 | 0.5217 | 0.6196 | 0.0000 | 49134 | collapse-detected | collapsed_small | 0 | 0 |  |

The guard-only salvage is technically safe in this simulation: lowering the collapse threshold to 0.295 or below changes only 02169 and leaves the 15 evolved cases untouched. But the gain is small: +0.0386 Dice on one case, about +0.0019 blended mean Dice across 20 cases. It also requires relaxing a core collapse threshold from 0.40 to about 0.295, which is not a tiny local tweak.

## Step 3: decision

Recommendation: CUT Stage 3B globally. Do not enable best-iterate selection.

For 02169, a narrow guard-only repair is possible in simulation, but the payoff is very small and it does not recover the genuinely best early contour. I would leave 02169 as a known method limit unless we explicitly want a separate, config-gated collapse-threshold experiment with the same regression gates.

Supporting files:

- `analysis/stage3b_postmortem_trajectory_detail.csv`
- `analysis/stage3b_postmortem_summary.csv`
- `analysis/stage3b_02169_collapse_sensitivity.csv`
- `analysis/stage3b_02169_collapse_sensitivity_detail.csv`
