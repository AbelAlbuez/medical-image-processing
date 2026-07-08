# Phase 2 Experiments

Phase 2 evaluates priors on the locked Stage-4 metric, never on pooled global
Dice. The locked scoring code is `phase2/metrics.py`.

## Canonical Architecture

Use `final_data/` as the paper-facing data layer. It contains one canonical CSV
per finding plus `DATA_DICTIONARY.md` and `SOURCE_MANIFEST.csv`. Intermediate
and superseded CSVs are retained under `_archive/` for auditability, but figures
and reports should prefer `final_data/` unless a prompt explicitly asks for an
archived diagnostic.

Publication outputs:

- `reports/`: academic markdown reports and paper scaffold.
- `figures/`: high-DPI paper/email figures and `CAPTIONS.md`.
- `brats_figures_bundle.zip`: packaged figure bundle.

Experiment subdirectories:

- `p1_spatial_atlas/`: leave-fold ET-occurrence atlas and spatial/location prior.
- `p2_shape_probe/`: component-shape separability probe.
- `p2_shape_prior/`: hard shape-proxy component filter.
- `p2_soft_shape_sweep/`: soft shape-proxy operating-point sweep.
- `p3_cubical_persistence/`: genuine GUDHI cubical H0 persistence diagnostic and count/rank prior.
- `surface_reconstruction/`: Open3D Poisson reconstruction and surface-distance analysis.
- `growth_metric/`: comparable baseline -> P1 -> P2b -> P3 growth tables.
- `four_regime/`: final four-regime master comparison and impossibility-chain table.
- `method_type/`: construction-type and per-parameter behavior summaries.

Files kept at the Phase 2 root are shared inputs or launch points:

- `metrics.py`: locked Stage-4/Phase-2 metric implementation.
- `generate_paper_figures.py`: regenerates the figure bundle from
  `final_data/`.
- P0/baseline target outputs are preserved in `final_data/` and archived copies.

Terminology warning: P2 shape features are shape-proxy geometry, not persistent
homology. P3 cubical H0 persistence is the only genuine topological computation
in this phase.
