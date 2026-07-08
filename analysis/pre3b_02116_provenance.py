from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.segmentation import morphological_chan_vese


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline import config, io_utils  # noqa: E402
from brats_pipeline.seg_et_pipeline import correr_pipeline_et  # noqa: E402
from brats_pipeline.seg_metrics import dice  # noqa: E402
from brats_pipeline import seg_spline_levelset as sls  # noqa: E402


CASE_ID = "BraTS-GLI-02116-100"
OUT_TABLE = ROOT / "analysis" / "pre3b_02116_provenance.csv"
OUT_MASKS = ROOT / "analysis" / "pre3b_02116_mask_inventory.csv"


def read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def clean(mod: str):
    return read(ROOT / "output" / "limpieza" / CASE_ID / f"{CASE_ID}-{mod}.nii.gz")


def raw(mod: str):
    return read(ROOT / "images" / CASE_ID / f"{CASE_ID}-{mod}.nii.gz")


def gt_img():
    return sitk.ReadImage(str(ROOT / "images" / CASE_ID / f"{CASE_ID}-seg.nii.gz"))


def gt_et(ref: sitk.Image):
    seg = sitk.Resample(
        sitk.Cast(gt_img(), sitk.sitkInt16),
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkInt16,
    )
    return (sitk.GetArrayFromImage(seg) == config.LABEL_ET)


def run_current(best_iter: bool, evidence_guard: bool, source: str) -> dict:
    config.ENABLE_BEST_ITERATE = best_iter
    config.ENABLE_EVIDENCE_GUARD = evidence_guard
    masks, gt, df = correr_pipeline_et(
        clean("t1c"),
        clean("t1n"),
        gt_img(),
        t1c_raw=raw("t1c"),
        t1n_raw=raw("t1n"),
        t2f=clean("t2f"),
        semilla_zyx=None,
        case_id=CASE_ID,
        auto_pct=90.0,
        sigma=0.5,
        verbose=False,
    )
    row = df[df["metodo"] == "variational_spline"].iloc[0]
    return {
        "source": source,
        "path": "",
        "mtime": "",
        "voxels": int(masks["variational_spline"].sum()),
        "gt_dice": float(dice(masks["variational_spline"], gt)),
        "guard_branch": row["guard_branch"],
        "guard_reason": row["guard_reason"],
        "notes": f"ENABLE_BEST_ITERATE={best_iter}; ENABLE_EVIDENCE_GUARD={evidence_guard}",
    }


def run_bypass_3b() -> dict:
    config.ENABLE_BEST_ITERATE = False
    config.ENABLE_EVIDENCE_GUARD = True
    t1c = clean("t1c")
    arr_t1c = io_utils.a_numpy(t1c).astype("float32")
    arr_t1c_raw = io_utils.a_numpy(raw("t1c")).astype("float32")
    arr_t1n_raw = io_utils.a_numpy(raw("t1n")).astype("float32")
    gt = gt_et(t1c)
    roi, mapa = sls.roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5)
    sl = sls._bbox(roi, margin=12, shape=mapa.shape)
    img = mapa[sl].astype("float32")
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    init = roi[sl].astype("uint8")
    sls._reset_morphsnakes_curvop("first")
    ls = morphological_chan_vese(
        img,
        num_iter=35,
        init_level_set=init,
        smoothing=3,
        lambda1=1.0,
        lambda2=1.0,
    ).astype("uint8")
    ls = sls._orientar_chanvese(ls, img)
    pred = roi * 0
    pred[sl] = ls
    pred = sls._post(pred, mapa, arr_t1c_raw > 0, init=roi, pct_realce=80.0)
    info = dict(sls.LAST_POST_INFO)
    return {
        "source": "2_current_code_3b_bypassed_direct_3a_path",
        "path": "",
        "mtime": "",
        "voxels": int(pred.sum()),
        "gt_dice": float(dice(pred, gt)),
        "guard_branch": info.get("branch", ""),
        "guard_reason": info.get("reason", ""),
        "notes": "Direct pre-3B variational_spline body; 3A evidence guard ON",
    }


def disk_mask_row(source: str, path: Path, notes: str) -> dict:
    t1c = clean("t1c")
    gt = gt_et(t1c)
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
    stat = path.stat()
    baseline = pd.read_csv(ROOT / "analysis" / "baseline" / "metricas_ET_baseline.csv")
    base_row = baseline[
        (baseline["case_id"] == CASE_ID)
        & (baseline["metodo"] == "variational_spline")
    ].iloc[0]
    return {
        "source": source,
        "path": str(path),
        "mtime": pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat(),
        "voxels": int(arr.sum()),
        "gt_dice": float(dice(arr, gt)),
        "guard_branch": base_row["guard_branch"],
        "guard_reason": base_row["guard_reason"],
        "notes": notes,
    }


def stage3a_report_row() -> dict:
    stage3a = pd.read_csv(ROOT / "analysis" / "stage3a_variational_spline.csv")
    row = stage3a[stage3a["case_id"] == CASE_ID].iloc[0]
    case_diff = pd.read_csv(ROOT / "analysis" / "stage3a_guard_case_diff.csv")
    diff = case_diff[case_diff["case_id"] == CASE_ID].iloc[0]
    return {
        "source": "5_stage3a_reported_result",
        "path": str(ROOT / "analysis" / "stage3a_variational_spline.csv"),
        "mtime": pd.Timestamp.fromtimestamp((ROOT / "analysis" / "stage3a_variational_spline.csv").stat().st_mtime).isoformat(),
        "voxels": int(diff["vol_pred_stage3a"]),
        "gt_dice": float(row["dice_ET_stage3a"]),
        "guard_branch": row["guard_branch_stage3a"],
        "guard_reason": row["guard_reason_stage3a"],
        "notes": "Reported artifact only; no mask file was written by 3A",
    }


def mask_inventory() -> pd.DataFrame:
    roots = [ROOT / "output", ROOT / "analysis", ROOT / "analysis" / "baseline"]
    rows = []
    seen = set()
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob(f"*{CASE_ID}*.nii*"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append(
                {
                    "path": str(path),
                    "mtime": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "bytes": path.stat().st_size,
                }
            )
    return pd.DataFrame(rows).sort_values("path")


def main() -> None:
    disk_path = ROOT / "output" / "segmentacion" / CASE_ID / f"{CASE_ID}-et_variational_spline.nii.gz"
    rows = [
        run_current(False, False, "1_current_code_best_off_gate0_env"),
        run_current(False, True, "1b_current_code_best_off_default_3a_guard_on"),
        run_bypass_3b(),
        disk_mask_row("3_output_segmentacion_mask_on_disk", disk_path, "Persisted NIfTI currently on disk"),
        disk_mask_row("4_frozen_mask_harness_compares_against", disk_path, "stage3b_validate.py reads this same path"),
        stage3a_report_row(),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLE, index=False)
    inv = mask_inventory()
    inv.to_csv(OUT_MASKS, index=False)
    print("provenance")
    print(out.to_string(index=False))
    print("\nmask_inventory")
    print(inv.to_string(index=False))


if __name__ == "__main__":
    main()
