# Phase 2 Experiments

Phase 2 evaluates priors on the locked Stage-4 metric, never on pooled global
Dice. The locked scoring code is `phase2/metrics.py`.

Subdirectories:

- `p1_spatial_atlas/`: leave-fold ET-occurrence atlas and spatial/location prior.
- `p2_shape_probe/`: component-shape separability probe.
- `p2_shape_prior/`: hard shape-proxy component filter.
- `p2_soft_shape_sweep/`: soft shape-proxy operating-point sweep.
- `p3_cubical_persistence/`: genuine GUDHI cubical H0 persistence diagnostic and count/rank prior.
- `surface_reconstruction/`: Open3D Poisson reconstruction and surface-distance analysis.
- `growth_metric/`: comparable baseline -> P1 -> P2b -> P3 growth tables.
- `four_regime/`: final four-regime master comparison and impossibility-chain table.
- `method_type/`: construction-type and per-parameter behavior summaries.

Files kept at the Phase 2 root are shared inputs (`metrics.py`, baseline target
CSVs, and P0 metric-lock outputs).

