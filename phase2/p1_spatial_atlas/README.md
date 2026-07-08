# P1 Spatial Atlas

Builds leave-fold ET-occurrence atlases from training folds and evaluates
location-based pre-filter, post-filter, and MAP modes on held-out folds.

Inputs: cohort manifest, cleaned cohort volumes, existing core-5 masks, and
`phase2/metrics.py`.

Verdict: location reduces flood only with a detection cost; confounds sit high
enough in atlas percentile space that the prior does not solve false positives.

