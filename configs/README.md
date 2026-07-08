# Configs

`pipeline.yaml` is the canonical runner configuration. It contains:

- the five-method core,
- the legacy/cut methods exposed by `--legacy`,
- seed and determinism-relevant settings,
- Stage 3A evidence-guard thresholds,
- preprocessing and segmentation parameters.

Changing values here changes numerical behavior and should be paired with the
appropriate regression and determinism checks.

