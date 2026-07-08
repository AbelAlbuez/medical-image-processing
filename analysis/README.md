# Analysis Artifacts

This directory contains audits, frozen baselines, and stage diagnostics.

- `baseline/` stores regression fixtures, post-fix baseline tables, determinism
  proof artifacts, and the baseline changelog.
- `stage4_metrics/` is the corrected case-type-aware metric suite used by
  Phase 2.
- `PRE_MORSE_GAP_REPORT.md` audits claim strength, leakage, mathematical rigor,
  and reproducibility before the Morse/topology decision.

Most files here are result artifacts. Treat them as read-only unless a staged
prompt explicitly asks to regenerate them.

