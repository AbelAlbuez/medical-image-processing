"""Trace the localized pre-Stage-1 nondeterminism without modifying src."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline import io_utils  # noqa: E402
from brats_pipeline.seg_metrics import dice  # noqa: E402
from brats_pipeline.seg_spline_levelset import (  # noqa: E402
    metodo_bspline,
    metodo_level_set,
    metodo_spline,
    metodo_variational_spline,
    roi_et_auto,
)


CASE = "BraTS-GLI-02108-100"
OUT = ROOT / "analysis" / "baseline" / "trace_pre_stage1_nondeterminism.csv"


def _read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def _clean(mod: str):
    return io_utils.a_numpy(_read(ROOT / "output" / "limpieza" / CASE / f"{CASE}-{mod}.nii.gz")).astype(np.float32)


def _raw(mod: str):
    return io_utils.a_numpy(_read(ROOT / "images" / CASE / f"{CASE}-{mod}.nii.gz")).astype(np.float32)


def _gt_et():
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "images" / CASE / f"{CASE}-seg.nii.gz")))
    return (seg == 3).astype(np.uint8)


def main() -> None:
    arr_t1c = _clean("t1c")
    arr_t1c_raw = _raw("t1c")
    arr_t1n_raw = _raw("t1n")
    gt = _gt_et()
    rows = []

    roi1, mapa1 = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5)
    roi2, mapa2 = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5)
    rows.append(
        {
            "component": "roi_et_auto",
            "run": "roi1_vs_roi2",
            "dice_vs_gt": dice(roi1, gt),
            "voxels": int(roi1.sum()),
            "equals_previous_same_component": bool(np.array_equal(roi1, roi2)),
            "mapa_max_abs_diff": float(np.max(np.abs(mapa1 - mapa2))),
            "equals_roi": True,
        }
    )

    methods = [
        ("level_set", metodo_level_set),
        ("variational_spline", metodo_variational_spline),
        ("bspline", metodo_bspline),
        ("spline", metodo_spline),
    ]
    for name, fn in methods:
        prev = None
        for run in (1, 2, 3):
            pred = fn(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5, roi=roi1, mapa=mapa1)
            rows.append(
                {
                    "component": name,
                    "run": run,
                    "dice_vs_gt": dice(pred, gt),
                    "voxels": int(pred.sum()),
                    "equals_previous_same_component": None if prev is None else bool(np.array_equal(pred, prev)),
                    "mapa_max_abs_diff": 0.0,
                    "equals_roi": bool(np.array_equal(pred, roi1)),
                }
            )
            prev = pred
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote={OUT}")


if __name__ == "__main__":
    main()
