# Removal Candidates For Review

No files were deleted in this cleanup pass. The items below look unused by import
or direct reference from the current runner/tests, but several may be historical
manual entry points. Review before deleting.

## Ambiguous Legacy / Manual Entry Points

| file | evidence | recommendation |
| --- | --- | --- |
| `scan_et.py` | Not imported by `run_all.py`, `src/`, `tests/`, `analysis/`, or `phase2/`; appears to be an exploratory scanner. | Keep until confirmed no thesis/report dependency. |
| `summarize.py` | Mentioned only by the old README workflow; not used by the config runner. | Replace with documented analysis outputs, then remove if no longer used. |
| `tune.py` | No current references found; likely old threshold tuning. | Candidate after verifying no saved report was produced from it. |
| `build_report.py` | No current runner/test references found. | Candidate if `analysis/report.md` is now produced elsewhere. |
| `make_report_figs.py` | No current runner/test references found. | Candidate if figure generation is superseded by analysis scripts. |

## Historical Analysis Scripts

These are not imported by the pipeline, but they document the audit trail and are
therefore not safe to delete without deciding how much provenance to preserve.

| file or pattern | evidence | recommendation |
| --- | --- | --- |
| `analysis/baseline/*.py` | One-off deterministic/baseline probes; not runner entry points. | Archive or keep as provenance. Do not delete blindly. |
| `analysis/stage3*.py` | One-off Stage 3 diagnostic scripts; not imported by tests. | Keep unless reports are fully reproducible from committed CSVs. |
| `analysis/stage4_metric_suite.py` | Stage 4 metric generator; superseded operationally by `phase2/metrics.py`, but useful for reproduction. | Keep unless Stage 4 README points to a newer canonical command. |
| `analysis/redundancy.py` | No current references found. | Candidate after checking old report generation. |

## Explicitly Not Candidates

- `run_all.py`
- `src/brats_pipeline/**`
- `tests/test_regression.py`
- `tests/test_determinism.py`
- `phase2/metrics.py`
- `configs/pipeline.yaml`
- `cohort/COHORT_MANIFEST.md`
- `cohort/COHORT_MANIFEST_selected.csv`
- `analysis/baseline/**` frozen CSV/JSON fixtures
- Any provenance JSON or run manifest
- Legacy/cut method implementations reachable through `--legacy`

