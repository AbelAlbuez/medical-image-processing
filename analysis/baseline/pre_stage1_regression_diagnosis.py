"""Pre-Stage-1 diagnosis: current-code rerun twice vs persisted baseline."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline.seg_et_pipeline import correr_pipeline_et  # noqa: E402


FIXTURE = ROOT / "analysis" / "baseline" / "regression_fixture.json"
OUT = ROOT / "analysis" / "baseline" / "pre_stage1_regression_diagnosis.csv"


def _read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def _clean(case_id: str, mod: str):
    return _read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")


def _raw(case_id: str, mod: str):
    return _read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")


def _gt(case_id: str):
    return sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz"))


def run_case(case_id: str) -> dict[str, float]:
    _, _, df = correr_pipeline_et(
        _clean(case_id, "t1c"),
        _clean(case_id, "t1n"),
        _gt(case_id),
        t1c_raw=_raw(case_id, "t1c"),
        t1n_raw=_raw(case_id, "t1n"),
        t2f=_clean(case_id, "t2f"),
        semilla_zyx=None,
        case_id=case_id,
        auto_pct=90.0,
        sigma=0.5,
        verbose=False,
    )
    return {row.metodo: float(row.dice_ET) for row in df.itertuples(index=False)}


def main() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tol = float(fixture["tolerance"])
    rows = []
    for case_id, spec in fixture["cases"].items():
        run1 = run_case(case_id)
        run2 = run_case(case_id)
        for method in sorted(spec["dice_ET"]):
            baseline = float(spec["dice_ET"][method])
            d1 = float(run1[method])
            d2 = float(run2[method])
            rows.append(
                {
                    "case": case_id,
                    "method": method,
                    "run1_dice": d1,
                    "run2_dice": d2,
                    "persisted_baseline_dice": baseline,
                    "run_abs_diff": abs(d1 - d2),
                    "baseline_abs_diff_run1": abs(d1 - baseline),
                    "run1_eq_run2_within_1e_3": abs(d1 - d2) <= tol,
                    "run1_eq_baseline_within_1e_3": abs(d1 - baseline) <= tol,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    nondeterministic = not out["run1_eq_run2_within_1e_3"].all()
    drift = out["run1_eq_run2_within_1e_3"].all() and not out["run1_eq_baseline_within_1e_3"].all()
    print(out.to_string(index=False))
    print()
    print(f"classification={'NONDETERMINISM' if nondeterministic else 'DRIFT' if drift else 'MATCH'}")
    print(f"wrote={OUT}")


if __name__ == "__main__":
    main()
