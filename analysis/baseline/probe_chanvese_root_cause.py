"""Probe Chan-Vese nondeterminism without modifying pipeline source."""
from __future__ import annotations

import hashlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.segmentation import morphsnakes
from skimage.segmentation import morphological_chan_vese


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline import io_utils  # noqa: E402
from brats_pipeline.seg_metrics import dice  # noqa: E402
from brats_pipeline.seg_spline_levelset import (  # noqa: E402
    _orientar_chanvese,
    _post,
    _bbox,
    roi_et_auto,
)


CASE = "BraTS-GLI-02108-100"
OUT = ROOT / "analysis" / "baseline" / "chanvese_input_hashes.csv"


def sha(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def reset_curvop() -> None:
    morphsnakes._curvop = morphsnakes._fcycle(  # noqa: SLF001
        [
            lambda u: morphsnakes.sup_inf(morphsnakes.inf_sup(u)),
            lambda u: morphsnakes.inf_sup(morphsnakes.sup_inf(u)),
        ]
    )


def _read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def _clean(mod: str):
    return io_utils.a_numpy(_read(ROOT / "output" / "limpieza" / CASE / f"{CASE}-{mod}.nii.gz")).astype(np.float32)


def _raw(mod: str):
    return io_utils.a_numpy(_read(ROOT / "images" / CASE / f"{CASE}-{mod}.nii.gz")).astype(np.float32)


def _gt_et():
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "images" / CASE / f"{CASE}-seg.nii.gz")))
    return (seg == 3).astype(np.uint8)


def run_cv(img: np.ndarray, init: np.ndarray, smoothing: int, reset: bool) -> np.ndarray:
    if reset:
        reset_curvop()
    ls = morphological_chan_vese(
        img,
        num_iter=35,
        init_level_set=init,
        smoothing=smoothing,
        lambda1=1.0,
        lambda2=1.0,
    ).astype(np.uint8)
    return _orientar_chanvese(ls, img)


def main() -> None:
    np.random.seed(0)
    random.seed(0)

    arr_t1c = _clean("t1c")
    arr_t1c_raw = _raw("t1c")
    arr_t1n_raw = _raw("t1n")
    gt = _gt_et()
    roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5)
    sl = _bbox(roi, margin=12, shape=mapa.shape)
    img = mapa[sl].astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    init = roi[sl].astype(np.uint8)
    cerebro = arr_t1c_raw > 0

    rows = []
    for scenario, reset in [
        ("seeded_no_reset", False),
        ("seeded_no_reset", False),
        ("curvop_reset", True),
        ("curvop_reset", True),
    ]:
        for smoothing, label in [(3, "variational_spline"), (2, "bspline_base")]:
            ls = run_cv(img, init, smoothing=smoothing, reset=reset)
            pred = np.zeros_like(mapa, dtype=np.uint8)
            pred[sl] = ls
            pct = 80.0 if label == "variational_spline" else 78.0
            post = _post(pred, mapa, cerebro, init=roi, pct_realce=pct)
            rows.append(
                {
                    "case_id": CASE,
                    "scenario": scenario,
                    "label": label,
                    "image_hash": sha(img),
                    "init_hash": sha(init),
                    "params": f"num_iter=35;smoothing={smoothing};lambda1=1.0;lambda2=1.0",
                    "raw_cv_hash": sha(ls),
                    "post_hash": sha(post),
                    "post_dice": dice(post, gt),
                    "post_voxels": int(post.sum()),
                    "equals_roi": bool(np.array_equal(post, roi)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"wrote={OUT}")


if __name__ == "__main__":
    main()
