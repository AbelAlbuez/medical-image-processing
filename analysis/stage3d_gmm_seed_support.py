from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline import io_utils  # noqa: E402
from brats_pipeline.seg_et_pipeline import metodo_gmm  # noqa: E402
from brats_pipeline.seg_metrics import dice  # noqa: E402
from brats_pipeline.seg_spline_levelset import MAX_SEED, _mapa_diferencia, _seed_diferencia  # noqa: E402

BASELINE = ROOT / "analysis" / "baseline" / "metricas_ET_baseline.csv"
OUT = ROOT / "analysis" / "stage3d_gmm_seed_support.csv"
TARGET_FAILURES = {"BraTS-GLI-00533-100", "BraTS-GLI-02078-100"}


def read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def clean(case_id: str, mod: str):
    return read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")


def raw(case_id: str, mod: str):
    return read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")


def centroid(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return pts.mean(axis=0).astype(float)


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    return float(np.linalg.norm(a - b))


def main() -> None:
    baseline = pd.read_csv(BASELINE)
    vs = baseline[baseline["metodo"] == "variational_spline"].copy()
    cases = sorted(vs["case_id"].unique())
    rows = []

    for case_id in cases:
        print(f"measuring {case_id}", flush=True)
        arr_t1c = io_utils.a_numpy(clean(case_id, "t1c")).astype(np.float32)
        arr_t1c_raw = io_utils.a_numpy(raw(case_id, "t1c")).astype(np.float32)
        arr_t1n_raw = io_utils.a_numpy(raw(case_id, "t1n")).astype(np.float32)
        cerebro = arr_t1c_raw > 0
        mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma=0.5)

        gmm = metodo_gmm(arr_t1c, n_comp=3)
        if gmm.any():
            gmm = ndimage.binary_fill_holes(gmm).astype(np.uint8)
        gmm_voxels = int(gmm.sum())

        vals_pos = mapa[cerebro & (mapa > 0)]
        brain_p90 = float(np.percentile(vals_pos, 90)) if vals_pos.size else float("nan")
        enh_support_mask = (mapa >= brain_p90) & cerebro if vals_pos.size else np.zeros_like(gmm, dtype=bool)
        gmm_vals = mapa[gmm > 0]
        gmm_support_frac = (
            float(np.logical_and(gmm > 0, enh_support_mask).sum() / gmm_voxels)
            if gmm_voxels else float("nan")
        )
        gmm_mean = float(gmm_vals.mean()) if gmm_vals.size else float("nan")
        gmm_p90 = float(np.percentile(gmm_vals, 90)) if gmm_vals.size else float("nan")
        gmm_max = float(gmm_vals.max()) if gmm_vals.size else float("nan")

        enh_blob = _seed_diferencia(mapa, cerebro)
        enh_blob_voxels = int(enh_blob.sum())
        overlap = int(np.logical_and(gmm > 0, enh_blob > 0).sum())
        gmm_enh_dice = float(dice(gmm, enh_blob))
        gmm_enh_centroid_distance = centroid_distance(centroid(gmm), centroid(enh_blob))

        branch = (
            "GMM-on-T1c primary"
            if 80 < gmm_voxels <= MAX_SEED
            else "enhancement-blob fallback"
            if enh_blob_voxels
            else "GMM fallback after empty enhancement"
        )

        vs_row = vs[vs["case_id"] == case_id].iloc[0]
        rows.append(
            {
                "case_id": case_id,
                "is_target_roi_failure": case_id in TARGET_FAILURES,
                "roi_branch_current": branch,
                "gmm_voxels": gmm_voxels,
                "brain_positive_mapa_p90": brain_p90,
                "gmm_seed_p90_support_frac": gmm_support_frac,
                "gmm_seed_mean_mapa": gmm_mean,
                "gmm_seed_p90_mapa": gmm_p90,
                "gmm_seed_max_mapa": gmm_max,
                "enhancement_blob_voxels": enh_blob_voxels,
                "gmm_enhancement_blob_overlap_voxels": overlap,
                "gmm_enhancement_blob_dice": gmm_enh_dice,
                "gmm_enhancement_blob_centroid_distance_vox": gmm_enh_centroid_distance,
                "variational_spline_dice": float(vs_row["dice_ET"]),
                "variational_spline_guard_branch": vs_row["guard_branch"],
                "variational_spline_guard_reason": vs_row["guard_reason"],
                "variational_spline_vol_pred": int(vs_row["vol_pred"]),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("gmm_seed_p90_support_frac", ascending=True).reset_index(drop=True)
    out["support_rank_low_to_high"] = np.arange(1, len(out) + 1)
    out.to_csv(OUT, index=False)

    cols = [
        "support_rank_low_to_high",
        "case_id",
        "is_target_roi_failure",
        "gmm_seed_p90_support_frac",
        "gmm_seed_mean_mapa",
        "gmm_enhancement_blob_dice",
        "gmm_enhancement_blob_centroid_distance_vox",
        "variational_spline_dice",
        "variational_spline_guard_branch",
        "variational_spline_guard_reason",
    ]
    print(out[cols].round(4).to_string(index=False))
    targets = out[out["is_target_roi_failure"]]
    non_targets = out[~out["is_target_roi_failure"]]
    max_target = float(targets["gmm_seed_p90_support_frac"].max())
    min_non_target = float(non_targets["gmm_seed_p90_support_frac"].min())
    print()
    print(f"max target support: {max_target:.6f}")
    print(f"min non-target support: {min_non_target:.6f}")
    print(f"clean gap: {min_non_target - max_target:.6f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
