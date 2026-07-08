from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage.segmentation import morphological_chan_vese


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline import config, io_utils  # noqa: E402
from brats_pipeline.seg_metrics import dice  # noqa: E402
from brats_pipeline.seg_spline_levelset import (  # noqa: E402
    _bbox,
    _orientar_chanvese,
    _post,
    _reset_morphsnakes_curvop,
    roi_et_auto,
)


SNAPSHOT_ITERS = {0, 5, 15, 25, 35}
STRUCT = ndimage.generate_binary_structure(3, 1)


def read_image(path: Path, pixel_type=sitk.sitkFloat32) -> sitk.Image:
    return sitk.ReadImage(str(path), pixel_type)


def clean(case_id: str, mod: str) -> sitk.Image:
    return read_image(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")


def raw(case_id: str, mod: str) -> sitk.Image:
    return read_image(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")


def gt_et(case_id: str, ref: sitk.Image) -> np.ndarray:
    seg = sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz"))
    seg = sitk.Resample(
        sitk.Cast(seg, sitk.sitkInt16),
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkInt16,
    )
    return (sitk.GetArrayFromImage(seg) == config.LABEL_ET).astype(np.uint8)


def gt_shape_stats(gt: np.ndarray) -> dict:
    vol = int(gt.sum())
    lab, n = ndimage.label(gt > 0, structure=STRUCT)
    comp_rows = []
    for label in range(1, n + 1):
        comp = lab == label
        size = int(comp.sum())
        if size < 20:
            continue
        coords = np.argwhere(comp)
        ext = coords.max(axis=0) - coords.min(axis=0) + 1
        comp_rows.append((size, ext))
    comp_rows.sort(reverse=True, key=lambda x: x[0])
    significant = [row for row in comp_rows if row[0] >= max(100, 0.01 * vol)]
    nonempty_areas = gt.sum(axis=(1, 2))
    nonempty_areas = nonempty_areas[nonempty_areas > 0]
    median_slice_area = float(np.median(nonempty_areas)) if len(nonempty_areas) else 0.0
    max_slice_area = int(nonempty_areas.max()) if len(nonempty_areas) else 0
    largest_frac = float(comp_rows[0][0] / vol) if comp_rows and vol else 0.0
    min_extent = int(min(comp_rows[0][1])) if comp_rows else 0
    return {
        "gt_et_voxels": vol,
        "gt_significant_components": len(significant),
        "gt_largest_component_fraction": largest_frac,
        "gt_largest_component_min_extent": min_extent,
        "gt_median_nonempty_slice_area": median_slice_area,
        "gt_max_slice_area": max_slice_area,
        "gt_small_lt_25k": bool(vol < 25_000),
        "gt_multifocal": bool(len(significant) > 1),
        "gt_thin": bool(min_extent <= 5 or median_slice_area < 200),
    }


def enhancement_stats(mapa: np.ndarray, roi: np.ndarray, gt: np.ndarray) -> dict:
    roi_vals = mapa[roi > 0]
    gt_vals = mapa[gt > 0]
    pos_roi = roi_vals[roi_vals > 0]
    return {
        "roi_mapa_mean": float(roi_vals.mean()) if roi_vals.size else 0.0,
        "roi_mapa_max": float(roi_vals.max()) if roi_vals.size else 0.0,
        "roi_mapa_p90": float(np.percentile(roi_vals, 90)) if roi_vals.size else 0.0,
        "roi_mapa_positive_fraction": float((roi_vals > 0).mean()) if roi_vals.size else 0.0,
        "roi_mapa_positive_mean": float(pos_roi.mean()) if pos_roi.size else 0.0,
        "gt_mapa_mean": float(gt_vals.mean()) if gt_vals.size else 0.0,
        "gt_mapa_max": float(gt_vals.max()) if gt_vals.size else 0.0,
    }


def run_trajectory(case_id: str, arr_t1c, arr_t1c_raw, arr_t1n_raw, roi, mapa, gt):
    cerebro = arr_t1c_raw > 0
    sl = _bbox(roi, margin=12, shape=mapa.shape)
    img = mapa[sl].astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    init = roi[sl].astype(np.uint8)
    snapshots: dict[int, np.ndarray] = {}
    callback_count = {"i": 0}

    def callback(u):
        i = callback_count["i"]
        if i in SNAPSHOT_ITERS:
            snapshots[i] = np.array(u, copy=True)
        callback_count["i"] += 1

    _reset_morphsnakes_curvop("first")
    morphological_chan_vese(
        img,
        num_iter=35,
        init_level_set=init,
        smoothing=3,
        lambda1=1.0,
        lambda2=1.0,
        iter_callback=callback,
    )

    rows = []
    for it in sorted(SNAPSHOT_ITERS):
        ls = snapshots[it].astype(np.uint8)
        ls = _orientar_chanvese(ls, img)
        raw_full = np.zeros_like(mapa, dtype=np.uint8)
        raw_full[sl] = ls
        post = _post(raw_full, mapa, cerebro, init=roi, pct_realce=80.0)
        rows.append(
            {
                "case_id": case_id,
                "iter": it,
                "raw_contour_voxels": int(raw_full.sum()),
                "raw_contour_dice": round(float(dice(raw_full, gt)), 4),
                "post_voxels": int(post.sum()),
                "post_dice": round(float(dice(post, gt)), 4),
            }
        )
    return rows


def classify(row: dict, trajectory: pd.DataFrame) -> str:
    final_raw = int(trajectory[trajectory["iter"] == 35]["raw_contour_voxels"].iloc[0])
    iter5 = int(trajectory[trajectory["iter"] == 5]["raw_contour_voxels"].iloc[0])
    roi_dice = float(row["roi_dice"])
    signal = float(row["roi_mapa_positive_mean"])
    best_raw_dice = float(trajectory["raw_contour_dice"].max())
    if roi_dice < 0.25 and best_raw_dice < 0.35:
        return "ROI-bad"
    if signal < 0.08 or float(row["roi_mapa_positive_fraction"]) < 0.55:
        return "signal-poor"
    if best_raw_dice >= roi_dice + 0.15:
        return "fixable-init"
    if iter5 > 0 and final_raw < 0.4 * int(row["roi_voxels"]):
        return "fixable-init"
    if row["guard_reason"] in {"empty_prediction", "collapsed_small"}:
        return "fixable-init"
    return "ROI-bad"


def main() -> None:
    metrics = pd.read_csv(ROOT / "output" / "tablas" / "metricas_ET.csv")
    fallback = metrics[
        (metrics["metodo"] == "variational_spline")
        & (metrics["guard_branch"] == "collapse-detected")
    ].copy()

    diagnosis_rows = []
    trajectory_rows = []
    for row in fallback.itertuples(index=False):
        case_id = row.case_id
        t1c_img = clean(case_id, "t1c")
        arr_t1c = io_utils.a_numpy(t1c_img).astype(np.float32)
        arr_t1c_raw = io_utils.a_numpy(raw(case_id, "t1c")).astype(np.float32)
        arr_t1n_raw = io_utils.a_numpy(raw(case_id, "t1n")).astype(np.float32)
        gt = gt_et(case_id, t1c_img)
        roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5)
        roi_d = float(dice(roi, gt))
        roi_gt_overlap = int(np.logical_and(roi > 0, gt > 0).sum())

        shape = gt_shape_stats(gt)
        stats = enhancement_stats(mapa, roi, gt)
        traj = run_trajectory(case_id, arr_t1c, arr_t1c_raw, arr_t1n_raw, roi, mapa, gt)
        trajectory_rows.extend(traj)
        traj_df = pd.DataFrame(traj)
        best_idx = traj_df["raw_contour_dice"].idxmax()
        best = traj_df.loc[best_idx]

        diag = {
            "case_id": case_id,
            "guard_reason": row.guard_reason,
            "fallback_dice": float(row.dice_ET),
            "fallback_voxels": int(row.vol_pred),
            "roi_voxels": int(roi.sum()),
            "roi_dice": round(roi_d, 4),
            "roi_gt_overlap_voxels": roi_gt_overlap,
            "roi_gt_recall": round(float(roi_gt_overlap / gt.sum()), 4) if gt.sum() else 0.0,
            "roi_precision": round(float(roi_gt_overlap / roi.sum()), 4) if roi.sum() else 0.0,
            **shape,
            **stats,
            "raw_best_iter": int(best["iter"]),
            "raw_best_dice": float(best["raw_contour_dice"]),
            "raw_best_voxels": int(best["raw_contour_voxels"]),
            "raw_final_voxels": int(
                traj_df.loc[traj_df["iter"] == 35, "raw_contour_voxels"].iloc[0]
            ),
            "raw_final_dice": float(
                traj_df.loc[traj_df["iter"] == 35, "raw_contour_dice"].iloc[0]
            ),
        }
        diag["classification"] = classify(diag, traj_df)
        diagnosis_rows.append(diag)

    diagnosis = pd.DataFrame(diagnosis_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    diagnosis.to_csv(ROOT / "analysis" / "varspline_fallback_diagnosis.csv", index=False)
    trajectories.to_csv(ROOT / "analysis" / "varspline_chanvese_trajectory.csv", index=False)

    print("diagnosis")
    print(diagnosis.to_string(index=False))
    print("\nclassification split")
    print(diagnosis["classification"].value_counts().to_string())
    print("\ntrajectory")
    print(trajectories.to_string(index=False))


if __name__ == "__main__":
    main()
