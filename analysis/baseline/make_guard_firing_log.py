"""Log deformable collapse-guard outcomes for all current cases."""
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
from brats_pipeline import seg_spline_levelset as sls  # noqa: E402


OUT = ROOT / "analysis" / "baseline" / "guard_firing_log.csv"


def _read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def _clean(case_id: str, mod: str):
    return io_utils.a_numpy(_read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")).astype(np.float32)


def _raw(case_id: str, mod: str):
    return io_utils.a_numpy(_read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")).astype(np.float32)


def _gt_et(case_id: str):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz")))
    return (seg == 3).astype(np.uint8)


def main() -> None:
    cases = sorted(p.name for p in (ROOT / "output" / "segmentacion").iterdir() if p.is_dir())
    methods = [
        ("level_set", sls.metodo_level_set),
        ("variational_spline", sls.metodo_variational_spline),
        ("bspline", sls.metodo_bspline),
        ("spline", sls.metodo_spline),
    ]
    rows = []
    for case_id in cases:
        arr_t1c = _clean(case_id, "t1c")
        arr_t1c_raw = _raw(case_id, "t1c")
        arr_t1n_raw = _raw(case_id, "t1n")
        gt = _gt_et(case_id)
        roi, mapa = sls.roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5)
        for method, fn in methods:
            pred = fn(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5, roi=roi, mapa=mapa)
            info = dict(sls.LAST_POST_INFO)
            rows.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "branch": info.get("branch", ""),
                    "reason": info.get("reason", ""),
                    "used_restriction": info.get("used_restriction", ""),
                    "init_voxels": info.get("init_voxels", ""),
                    "raw_voxels": info.get("raw_voxels", ""),
                    "restricted_voxels": info.get("restricted_voxels", ""),
                    "final_voxels": info.get("final_voxels", int(pred.sum())),
                    "dice_ET": dice(pred, gt),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.groupby(["method", "branch", "reason"]).size().reset_index(name="n").to_string(index=False))
    print(f"wrote={OUT}")


if __name__ == "__main__":
    main()
