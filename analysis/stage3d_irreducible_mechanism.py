from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline import config, io_utils  # noqa: E402
from brats_pipeline.seg_et_pipeline import metodo_gmm  # noqa: E402
from brats_pipeline.seg_metrics import dice  # noqa: E402
from brats_pipeline.seg_spline_levelset import _mapa_diferencia, _seed_diferencia  # noqa: E402

TARGETS = ["BraTS-GLI-00533-100", "BraTS-GLI-02078-100"]
OUT = ROOT / "analysis" / "stage3d_irreducible_mechanism.csv"
FIG_DIR = ROOT / "analysis" / "stage3d_irreducible_overlays"
FIG_DIR.mkdir(exist_ok=True)
VISUAL_NOTES = {
    "BraTS-GLI-00533-100": (
        "resection-margin / cavity-adjacent non-tumor enhancement",
        "Overlay inspection: red GMM seed is central/anterior-periventricular and touches the RC neighborhood; "
        "green ET is lateral and separate. Pattern is compatible with cavity-adjacent or vascular/choroid-like "
        "non-tumor enhancement, not contralateral tumor."
    ),
    "BraTS-GLI-02078-100": (
        "midline vascular/dural-like non-tumor enhancement near resection region",
        "Overlay inspection: red GMM seed is centered near midline/anterior-inferior bright linear enhancement and "
        "is separated from the lateral green ET and blue RC. Pattern is compatible with vascular/dural or other "
        "non-tumor enhancement near the resection region, not contralateral tumor."
    ),
}


def read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def clean(case_id: str, mod: str):
    return read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")


def raw(case_id: str, mod: str):
    return read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")


def seg(case_id: str):
    return sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz"))


def centroid(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return pts.mean(axis=0).astype(float)


def bbox(mask: np.ndarray):
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return None
    return pts.min(axis=0), pts.max(axis=0)


def bbox_text(mask: np.ndarray) -> str:
    b = bbox(mask)
    if b is None:
        return "empty"
    lo, hi = b
    return f"z{lo[0]}:{hi[0]},y{lo[1]}:{hi[1]},x{lo[2]}:{hi[2]}"


def centroid_text(c: np.ndarray) -> str:
    if np.isnan(c).any():
        return ""
    return ";".join(f"{v:.2f}" for v in c)


def normalized_centroid(c: np.ndarray, brain: np.ndarray) -> np.ndarray:
    b = bbox(brain)
    if b is None or np.isnan(c).any():
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    lo, hi = b
    denom = np.maximum(hi - lo, 1)
    return (c - lo) / denom


def distance_vox(a: np.ndarray, b: np.ndarray) -> float:
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    return float(np.linalg.norm(a - b))


def min_distance(src: np.ndarray, dst: np.ndarray) -> float:
    if not src.any() or not dst.any():
        return float("nan")
    if np.logical_and(src, dst).any():
        return 0.0
    dt = ndimage.distance_transform_edt(~dst.astype(bool))
    return float(dt[src.astype(bool)].min())


def largest_component_fraction(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    lab, n = ndimage.label(mask > 0, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return 0.0
    sizes = ndimage.sum(mask > 0, lab, range(1, n + 1))
    return float(np.max(sizes) / mask.sum())


def norm2d(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a[a > 0], [1, 99]) if np.any(a > 0) else (float(a.min()), float(a.max()))
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def overlay_rgb(base: np.ndarray, gmm: np.ndarray, gt: np.ndarray, rc: np.ndarray, enh: np.ndarray) -> np.ndarray:
    rgb = np.dstack([base, base, base]).astype(float)
    rgb[gmm > 0] = 0.45 * rgb[gmm > 0] + np.array([1.0, 0.05, 0.05]) * 0.55
    rgb[gt > 0] = 0.35 * rgb[gt > 0] + np.array([0.05, 1.0, 0.05]) * 0.65
    rgb[rc > 0] = 0.45 * rgb[rc > 0] + np.array([0.05, 0.35, 1.0]) * 0.55
    edge = ndimage.binary_dilation(enh > 0, iterations=1) ^ (enh > 0)
    rgb[edge] = np.array([1.0, 0.9, 0.05])
    return np.clip(rgb, 0, 1)


def render(case_id: str, t1c: np.ndarray, mapa: np.ndarray, gmm: np.ndarray,
           gt: np.ndarray, rc: np.ndarray, enh_blob: np.ndarray) -> str:
    cg = centroid(gmm).round().astype(int)
    ct = centroid(gt).round().astype(int)
    z_vals = [int(cg[0]), int(ct[0])]
    y_vals = [int(cg[1]), int(ct[1])]
    x_vals = [int(cg[2]), int(ct[2])]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    specs = [
        ("axial @ GMM z", "z", z_vals[0]),
        ("axial @ GT z", "z", z_vals[1]),
        ("coronal @ GMM y", "y", y_vals[0]),
        ("coronal @ GT y", "y", y_vals[1]),
        ("sagittal @ GMM x", "x", x_vals[0]),
        ("sagittal @ GT x", "x", x_vals[1]),
    ]
    base_t1c = norm2d(t1c)
    for ax, (title, axis, idx) in zip(axes.ravel(), specs):
        if axis == "z":
            sl = (idx, slice(None), slice(None))
        elif axis == "y":
            sl = (slice(None), idx, slice(None))
        else:
            sl = (slice(None), slice(None), idx)
        rgb = overlay_rgb(base_t1c[sl], gmm[sl], gt[sl], rc[sl], enh_blob[sl])
        if axis != "z":
            rgb = np.rot90(rgb)
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{case_id}: red=GMM seed, green=GT ET, blue=RC, yellow=edge enhancement blob")
    fig.tight_layout()
    out = FIG_DIR / f"{case_id}_gmm_gt_rc_overlay.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def classify(case_id: str, gmm: np.ndarray, gt: np.ndarray, rc: np.ndarray,
             brain: np.ndarray, mapa: np.ndarray) -> tuple[str, str]:
    c_gmm = centroid(gmm)
    c_gt = centroid(gt)
    c_rc = centroid(rc)
    n = normalized_centroid(c_gmm, brain)
    gmm_rc_overlap = int(np.logical_and(gmm, rc).sum())
    d_gmm_rc = min_distance(gmm, rc)
    d_gmm_gt = min_distance(gmm, gt)
    gmm_mean = float(mapa[gmm > 0].mean()) if gmm.any() else float("nan")
    gt_mean = float(mapa[gt > 0].mean()) if gt.any() else float("nan")

    if rc.any() and (gmm_rc_overlap > 0 or d_gmm_rc <= 2.0):
        label = "resection-margin / cavity-adjacent non-tumor enhancement"
        rationale = (
            f"GMM is disjoint from ET but overlaps/touches RC "
            f"(overlap={gmm_rc_overlap}, min_dist={d_gmm_rc:.1f} vox); "
            f"mapa mean inside GMM {gmm_mean:.3f} exceeds GT mean {gt_mean:.3f}."
        )
    elif rc.any() and distance_vox(c_gmm, c_rc) < distance_vox(c_gmm, c_gt):
        label = "resection-region non-tumor enhancement"
        rationale = (
            f"GMM centroid is closer to RC than ET "
            f"(GMM-RC centroid {distance_vox(c_gmm, c_rc):.1f} vox, "
            f"GMM-ET centroid {distance_vox(c_gmm, c_gt):.1f} vox); "
            f"brain-normalized centroid z/y/x={centroid_text(n)}."
        )
    elif 0.40 <= n[2] <= 0.60:
        label = "midline/choroid-plexus-like non-tumor enhancement"
        rationale = (
            f"GMM is central in x (brain-normalized x={n[2]:.2f}) and disjoint from ET; "
            f"mapa mean inside GMM {gmm_mean:.3f} exceeds GT mean {gt_mean:.3f}."
        )
    else:
        label = "other bright non-tumor enhancement"
        rationale = (
            f"GMM is disjoint from ET (min distance {d_gmm_gt:.1f} vox) and lies at "
            f"brain-normalized z/y/x={centroid_text(n)}; "
            f"mapa mean inside GMM {gmm_mean:.3f} exceeds GT mean {gt_mean:.3f}."
        )
    return label, rationale


def main() -> None:
    rows = []
    for case_id in TARGETS:
        arr_t1c = io_utils.a_numpy(clean(case_id, "t1c")).astype(np.float32)
        arr_t1c_raw = io_utils.a_numpy(raw(case_id, "t1c")).astype(np.float32)
        arr_t1n_raw = io_utils.a_numpy(raw(case_id, "t1n")).astype(np.float32)
        seg_arr = sitk.GetArrayFromImage(seg(case_id))
        gt = seg_arr == config.LABEL_ET
        rc = seg_arr == 4
        brain = arr_t1c_raw > 0
        mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma=0.5)
        gmm = metodo_gmm(arr_t1c, n_comp=3)
        if gmm.any():
            gmm = ndimage.binary_fill_holes(gmm).astype(np.uint8)
        enh_blob = _seed_diferencia(mapa, brain)
        c_gmm = centroid(gmm)
        c_gt = centroid(gt)
        c_rc = centroid(rc)
        n_gmm = normalized_centroid(c_gmm, brain)
        mechanism, rationale = classify(case_id, gmm > 0, gt, rc, brain, mapa)
        fig_path = render(case_id, arr_t1c_raw, mapa, gmm > 0, gt, rc, enh_blob > 0)
        if case_id in VISUAL_NOTES:
            mechanism, visual_note = VISUAL_NOTES[case_id]
            rationale = f"{rationale} {visual_note}"
        rows.append({
            "case_id": case_id,
            "mechanism_class": mechanism,
            "mechanism_rationale": rationale,
            "gmm_centroid_zyx": centroid_text(c_gmm),
            "gt_et_centroid_zyx": centroid_text(c_gt),
            "rc_centroid_zyx": centroid_text(c_rc),
            "gmm_centroid_brain_norm_zyx": centroid_text(n_gmm),
            "gmm_gt_centroid_distance_vox": distance_vox(c_gmm, c_gt),
            "gmm_rc_centroid_distance_vox": distance_vox(c_gmm, c_rc),
            "gmm_to_gt_min_distance_vox": min_distance(gmm > 0, gt),
            "gmm_to_rc_min_distance_vox": min_distance(gmm > 0, rc),
            "gmm_rc_overlap_voxels": int(np.logical_and(gmm > 0, rc).sum()),
            "gmm_rc_dice": dice(gmm > 0, rc),
            "gmm_gt_dice": dice(gmm > 0, gt),
            "gmm_voxels": int(gmm.sum()),
            "gt_et_voxels": int(gt.sum()),
            "rc_voxels": int(rc.sum()),
            "gmm_largest_component_fraction": largest_component_fraction(gmm),
            "gmm_mapa_mean": float(mapa[gmm > 0].mean()) if gmm.any() else np.nan,
            "gt_et_mapa_mean": float(mapa[gt].mean()) if gt.any() else np.nan,
            "gmm_bbox_zyx": bbox_text(gmm > 0),
            "gt_et_bbox_zyx": bbox_text(gt),
            "rc_bbox_zyx": bbox_text(rc),
            "overlay_png": fig_path,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out[[
        "case_id", "mechanism_class", "gmm_centroid_zyx",
        "gmm_centroid_brain_norm_zyx", "gmm_gt_centroid_distance_vox",
        "gmm_to_rc_min_distance_vox", "gmm_rc_overlap_voxels",
        "gmm_mapa_mean", "gt_et_mapa_mean", "overlay_png",
    ]].round(4).to_string(index=False))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
