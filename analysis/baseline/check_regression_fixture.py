"""Run the Stage 0 regression fixture and write an observed-vs-baseline diff."""
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
OUT = ROOT / "analysis" / "baseline" / "regression_observed_vs_baseline.csv"


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
    rows = []
    for case_id, spec in fixture["cases"].items():
        observed = run_case(case_id)
        for method, expected in spec["dice_ET"].items():
            obs = observed[method]
            rows.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "baseline_dice": expected,
                    "rerun_dice": obs,
                    "delta": obs - expected,
                    "abs_delta": abs(obs - expected),
                    "within_1e_3": abs(obs - expected) <= float(fixture["tolerance"]),
                }
            )
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
