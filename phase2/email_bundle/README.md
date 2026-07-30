# BraTS 2024 GLI ET Segmentation Study Bundle

This bundle contains derived figures, reports, and summary tables from a classical post-treatment BraTS-2024 GLI enhancing-tumor (ET, label 3) segmentation study. The headline finding is a two-axis impossibility result: methods operating only on the enhancement scalar struggle both to detect true post-treatment ET and to restrain false-positive enhancement, and post-hoc spatial, shape-proxy, and cubical-persistence priors do not reliably resolve that tradeoff.

## What Is Included

- `figures/`: publication/email PNG figures plus `CAPTIONS.md`.
- `reports/MAIN_FINDINGS.md`: academic summary of the four-regime comparison, growth metric, and surface analysis.
- `reports/CAVEATS_AND_GAPS.md`: robustness caveats, leakage risks, small-n limitations, and reproducibility scope.
- `reports/NEXT_STEPS.md`: how the findings motivate integrated learned/topological methods.
- `reports/PAPER_OUTLINE.md`: section-by-section paper scaffold with robustness labels.
- `data/four_regime_master_comparison.csv`: method-level master comparison across intensity, spatial/location, shape-proxy geometry, and cubical persistent homology regimes.
- `data/four_regime_findings.md`: prose interpretation of the impossibility chain.
- `data/growth_metric_findings.md`: baseline to P1/P2b/P3 behavior across the locked Stage-4 metric axes.
- `surface_reconstruction/SURFACE_RECONSTRUCTION_REPORT.md`: Poisson reconstruction and surface-distance analysis.
- `surface_reconstruction/*.png`: key derived surface renders for irreducible/confound-heavy cases.

## Sensitive Data Check

This email bundle is intentionally limited to derived reports, figures, and aggregate/summary CSVs. It does not include raw NIfTI imaging, cleaned volumes, segmentation masks, model checkpoints, or gated BraTS training data. BraTS case identifiers may appear in figures or reports as de-identified challenge IDs used for scientific traceability.
