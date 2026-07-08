"""Run one case through current code and print per-method mask hashes as JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline.seg_et_pipeline import correr_pipeline_et  # noqa: E402


def hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def _clean(case_id: str, mod: str):
    return _read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")


def _raw(case_id: str, mod: str):
    return _read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")


def _gt(case_id: str):
    return sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz"))


def run_case(case_id: str) -> dict:
    masks, _, df = correr_pipeline_et(
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
    dice_by_method = {row.metodo: float(row.dice_ET) for row in df.itertuples(index=False)}
    return {
        "case_id": case_id,
        "methods": {
            name: {
                "hash": hash_array(mask.astype(np.uint8)),
                "dice": dice_by_method[name],
                "voxels": int(mask.sum()),
            }
            for name, mask in masks.items()
            if not name.startswith("_")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    args = parser.parse_args()
    print(json.dumps(run_case(args.case_id), sort_keys=True))


if __name__ == "__main__":
    main()
