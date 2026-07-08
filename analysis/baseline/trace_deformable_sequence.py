"""Trace deformable driver sequence for the nondeterministic fixture case."""
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
    correr_spline_levelset,
    metodo_bspline,
    metodo_level_set,
    metodo_spline,
    metodo_variational_spline,
    roi_et_auto,
)


CASE = "BraTS-GLI-02108-100"
OUT = ROOT / "analysis" / "baseline" / "trace_deformable_sequence.csv"


def _read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def _clean(mod: str):
    return io_utils.a_numpy(_read(ROOT / "output" / "limpieza" / CASE / f"{CASE}-{mod}.nii.gz")).astype(np.float32)


def _raw(mod: str):
    return io_utils.a_numpy(_read(ROOT / "images" / CASE / f"{CASE}-{mod}.nii.gz")).astype(np.float32)


def _gt_et():
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "images" / CASE / f"{CASE}-seg.nii.gz")))
    return (seg == 3).astype(np.uint8)


def summarize(batch: str, run: int, masks: dict, gt: np.ndarray, rows: list) -> None:
    roi, _ = roi_et_auto(_clean("t1c"), _raw("t1c"), _raw("t1n"), sigma=0.5)
    for method in ["level_set", "variational_spline", "bspline", "spline"]:
        pred = masks[method]
        rows.append(
            {
                "batch": batch,
                "run": run,
                "method": method,
                "dice_vs_gt": dice(pred, gt),
                "voxels": int(pred.sum()),
                "equals_roi": bool(np.array_equal(pred, roi)),
            }
        )


def manual_sequence(arr_t1c, arr_t1c_raw, arr_t1n_raw):
    roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5)
    return {
        "level_set": metodo_level_set(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5, roi=roi, mapa=mapa),
        "variational_spline": metodo_variational_spline(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5, roi=roi, mapa=mapa),
        "bspline": metodo_bspline(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5, roi=roi, mapa=mapa),
        "spline": metodo_spline(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5, roi=roi, mapa=mapa),
    }


def main() -> None:
    arr_t1c = _clean("t1c")
    arr_t1c_raw = _raw("t1c")
    arr_t1n_raw = _raw("t1n")
    gt = _gt_et()
    rows = []
    for run in (1, 2, 3):
        summarize(
            "correr_spline_levelset",
            run,
            correr_spline_levelset(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5, verbose=False),
            gt,
            rows,
        )
    for run in (1, 2, 3):
        summarize("manual_sequence", run, manual_sequence(arr_t1c, arr_t1c_raw, arr_t1n_raw), gt, rows)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"wrote={OUT}")


if __name__ == "__main__":
    main()
