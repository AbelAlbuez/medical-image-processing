"""Evaluate Poisson surface reconstruction fidelity for GT and predictions.

Inputs are existing GT masks, baseline core-5 masks, and cohort metadata. Outputs
are reconstruction-error floors, surface-distance summaries, and diagnostic
figures in ``phase2/surface_reconstruction``.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_ROOT = ROOT / "analysis" / "pre_cohort_run" / "cohort_100_core"
CLEAN_ROOT = RUN_ROOT / "limpieza"
SEG_ROOT = RUN_ROOT / "segmentacion"
MANIFEST = ROOT / "cohort" / "COHORT_MANIFEST_selected.csv"
STAGE4_PRESENT = ROOT / "analysis" / "stage4_metrics" / "stage4_present_by_vol_bin.csv"
STAGE4_CASE = ROOT / "analysis" / "stage4_metrics" / "stage4_case_metrics.csv"
IRREDUCIBLE = ROOT / "analysis" / "stage3d_irreducible_mechanism.csv"
OLD20_DATA_ROOT = ROOT / "images"
OLD20_MASK_ROOT = ROOT / "analysis" / "stage5_runner_validation" / "run_20260706_220739" / "segmentacion"
OUT = ROOT / "phase2" / "surface_reconstruction"
SURF_DIR = OUT / "surface_points"
FIG_DIR = OUT / "figures"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
VOL_ORDER = ["small", "medium", "large"]
STRUCT26 = np.ones((3, 3, 3), dtype=bool)
RNG_SEED = 0

MAX_BOUNDARY_POINTS = 8_000
TARGET_SURFACE_POINTS = 8_000
POISSON_DEPTH = 7
MIN_COMPONENT_SIZE = 10
MAX_RENDER_POINTS = 6_000


def read_image(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def as_et(seg_arr: np.ndarray) -> np.ndarray:
    return np.round(seg_arr).astype(np.int16) == 3


def pred_path(case_id: str, method: str, root: Path = SEG_ROOT) -> Path:
    return root / case_id / f"{case_id}-et_{method}.nii.gz"


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=STRUCT26, border_value=0)
    return mask & ~eroded


def metric_component_count(mask: np.ndarray) -> int:
    labels, n = ndimage.label(mask.astype(bool), structure=STRUCT26)
    if n == 0:
        return 0
    sizes = np.bincount(labels.ravel())
    return int(sum(sizes[label] >= MIN_COMPONENT_SIZE for label in range(1, n + 1)))


def surface_area_proxy(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    dz = np.abs(np.diff(padded, axis=0)).sum()
    dy = np.abs(np.diff(padded, axis=1)).sum()
    dx = np.abs(np.diff(padded, axis=2)).sum()
    return float(dz + dy + dx)


def mask_geometry(mask: np.ndarray) -> Dict[str, float]:
    vol = int(mask.sum())
    surface = surface_area_proxy(mask)
    n_comp = metric_component_count(mask)
    if vol == 0 or surface == 0:
        return {
            "gt_vox": vol,
            "gt_components": n_comp,
            "surface_area_proxy": surface,
            "volume_surface_ratio": np.nan,
            "surface_volume_ratio": np.nan,
            "isoperimetric_quotient": np.nan,
            "thin_shell_proxy": np.nan,
            "multifocal": bool(n_comp > 1),
        }
    return {
        "gt_vox": vol,
        "gt_components": n_comp,
        "surface_area_proxy": surface,
        "volume_surface_ratio": float(vol / surface),
        "surface_volume_ratio": float(surface / vol),
        "isoperimetric_quotient": float(36.0 * np.pi * (vol**2) / (surface**3)),
        "thin_shell_proxy": float(surface / max(1.0, vol ** (2.0 / 3.0))),
        "multifocal": bool(n_comp > 1),
    }


def coords_from_mask(mask: np.ndarray, spacing_xyz: Iterable[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    boundary = mask_boundary(mask)
    coords_zyx = np.argwhere(boundary)
    if coords_zyx.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    spacing = np.asarray(tuple(spacing_xyz), dtype=np.float32)
    # SimpleITK spacing is xyz; numpy coordinates are zyx. Store points in xyz.
    points = coords_zyx[:, [2, 1, 0]].astype(np.float32) * spacing[None, :]
    return points


def deterministic_sample(points: np.ndarray, max_points: int, seed_parts: Tuple[str, ...]) -> np.ndarray:
    if len(points) <= max_points:
        return points
    digest = hashlib.sha256("|".join(seed_parts).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(idx)]


def reconstruct_points(mask: np.ndarray,
                       spacing_xyz: Iterable[float],
                       cache_path: Path,
                       cache_key: Tuple[str, ...]) -> Tuple[np.ndarray, Dict[str, object]]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        points = data["points"].astype(np.float32)
        meta = {key: data[key].item() if data[key].shape == () else data[key].tolist()
                for key in data.files if key != "points"}
        return points, meta

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_points_all = coords_from_mask(mask, spacing_xyz)
    meta: Dict[str, object] = {
        "input_voxels": int(mask.sum()),
        "boundary_points_total": int(len(boundary_points_all)),
        "poisson_depth": POISSON_DEPTH,
        "status": "ok",
        "fallback_used": False,
    }
    if len(boundary_points_all) < 4:
        points = boundary_points_all.astype(np.float32)
        meta["status"] = "too_few_boundary_points"
        np.savez_compressed(cache_path, points=points, **meta)
        return points, meta

    boundary_points = deterministic_sample(boundary_points_all, MAX_BOUNDARY_POINTS, cache_key)
    meta["boundary_points_used"] = int(len(boundary_points))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(boundary_points.astype(np.float64))
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30)
        )
        center = boundary_points.mean(axis=0)
        pcd.orient_normals_towards_camera_location(center + np.array([0.0, 0.0, 1_000.0]))
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=POISSON_DEPTH, width=0, scale=1.1, linear_fit=False
        )
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.05, bbox.get_center())
        mesh = mesh.crop(bbox)
        if len(densities) == len(mesh.vertices) and len(mesh.vertices) > 100:
            dens = np.asarray(densities)
            keep = dens >= np.quantile(dens, 0.02)
            if len(keep) == len(mesh.vertices):
                mesh.remove_vertices_by_mask(~keep)
        if len(mesh.triangles) > 0 and len(mesh.vertices) > 0:
            sampled = mesh.sample_points_uniformly(number_of_points=TARGET_SURFACE_POINTS)
            points = np.asarray(sampled.points, dtype=np.float32)
            meta["mesh_vertices"] = int(len(mesh.vertices))
            meta["mesh_triangles"] = int(len(mesh.triangles))
        else:
            points = boundary_points.astype(np.float32)
            meta["fallback_used"] = True
            meta["status"] = "empty_mesh_fallback_boundary"
    except Exception as exc:
        points = boundary_points.astype(np.float32)
        meta["fallback_used"] = True
        meta["status"] = f"poisson_failed_boundary_fallback:{type(exc).__name__}"

    np.savez_compressed(cache_path, points=points, **meta)
    return points, meta


def pointset_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if len(a) == 0 or len(b) == 0:
        return {
            "hd95": np.nan,
            "asd": np.nan,
            "chamfer": np.nan,
            "a_to_b_mean": np.nan,
            "b_to_a_mean": np.nan,
        }
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    d_ab = tree_b.query(a, k=1, workers=-1)[0]
    d_ba = tree_a.query(b, k=1, workers=-1)[0]
    sym = np.concatenate([d_ab, d_ba])
    return {
        "hd95": float(np.percentile(sym, 95)),
        "asd": float(sym.mean()),
        "chamfer": float(0.5 * (np.mean(d_ab**2) + np.mean(d_ba**2))),
        "a_to_b_mean": float(d_ab.mean()),
        "b_to_a_mean": float(d_ba.mean()),
    }


def gt_surface_points(case_id: str, vol_bin: str) -> Tuple[np.ndarray, np.ndarray, sitk.Image, Dict[str, object]]:
    seg, img = read_image(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
    gt = as_et(seg)
    cache = SURF_DIR / "cohort" / case_id / "GT_ET_poisson_points.npz"
    points, meta = reconstruct_points(gt, img.GetSpacing(), cache, (case_id, "GT_ET"))
    return gt, points, img, meta


def prediction_surface_points(case_id: str, method: str, spacing_xyz: Iterable[float]) -> Tuple[np.ndarray, Dict[str, object]]:
    pred, _ = read_image(pred_path(case_id, method))
    pred = pred > 0
    cache = SURF_DIR / "cohort" / case_id / f"{method}_poisson_points.npz"
    return reconstruct_points(pred, spacing_xyz, cache, (case_id, method))


def reconstruction_floor(process: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in process.reset_index(drop=True).iterrows():
        case_id = str(row["case_id"])
        vol_bin = str(row["vol_bin"])
        print(f"[surface floor] {idx + 1:03d}/{len(process)} {case_id}", flush=True)
        gt, recon, img, meta = gt_surface_points(case_id, vol_bin)
        raw_boundary = coords_from_mask(gt, img.GetSpacing())
        metrics = pointset_metrics(recon, raw_boundary)
        rows.append({
            "case_id": case_id,
            "vol_bin": vol_bin,
            "focality": row["focality"],
            **mask_geometry(gt),
            "recon_points": int(len(recon)),
            "raw_boundary_points": int(len(raw_boundary)),
            "recon_status": meta.get("status", ""),
            "recon_fallback_used": bool(meta.get("fallback_used", False)),
            "floor_hd95": metrics["hd95"],
            "floor_asd": metrics["asd"],
            "floor_chamfer": metrics["chamfer"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "surface_gt_reconstruction_floor_cases.csv", index=False)
    return out


def floor_summary(floor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vol in VOL_ORDER:
        sub = floor[floor["vol_bin"].eq(vol)]
        rows.append({
            "vol_bin": vol,
            "n": int(len(sub)),
            "floor_hd95_mean": float(sub["floor_hd95"].mean()),
            "floor_hd95_median": float(sub["floor_hd95"].median()),
            "floor_asd_mean": float(sub["floor_asd"].mean()),
            "floor_asd_median": float(sub["floor_asd"].median()),
            "floor_chamfer_median": float(sub["floor_chamfer"].median()),
            "fallback_count": int(sub["recon_fallback_used"].sum()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "surface_gt_reconstruction_floor_by_stratum.csv", index=False)
    return out


def floor_correlations(floor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = [
        "gt_vox", "gt_components", "surface_area_proxy", "volume_surface_ratio",
        "surface_volume_ratio", "isoperimetric_quotient", "thin_shell_proxy",
    ]
    for metric in ["floor_hd95", "floor_asd", "floor_chamfer"]:
        for pred in predictors:
            sub = floor[[metric, pred]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 4:
                rho = np.nan
                p = np.nan
            else:
                rho, p = spearmanr(sub[pred], sub[metric])
            rows.append({
                "floor_metric": metric,
                "geometry_predictor": pred,
                "n": int(len(sub)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "surface_floor_geometry_correlations.csv", index=False)
    return out


def prediction_surface_fidelity(process: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in process.reset_index(drop=True).iterrows():
        case_id = str(row["case_id"])
        vol_bin = str(row["vol_bin"])
        gt, gt_recon, img, gt_meta = gt_surface_points(case_id, vol_bin)
        for method in METHODS:
            print(f"[surface pred] {idx + 1:03d}/{len(process)} {case_id} {method}", flush=True)
            pred_recon, pred_meta = prediction_surface_points(case_id, method, img.GetSpacing())
            metrics = pointset_metrics(pred_recon, gt_recon)
            rows.append({
                "case_id": case_id,
                "vol_bin": vol_bin,
                "focality": row["focality"],
                "method": method,
                "pred_recon_points": int(len(pred_recon)),
                "gt_recon_points": int(len(gt_recon)),
                "pred_recon_status": pred_meta.get("status", ""),
                "pred_recon_fallback_used": bool(pred_meta.get("fallback_used", False)),
                "surface_hd95": metrics["hd95"],
                "surface_asd": metrics["asd"],
                "surface_chamfer": metrics["chamfer"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "surface_prediction_fidelity_cases.csv", index=False)
    return out


def prediction_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stage4 = pd.read_csv(STAGE4_PRESENT)
    for (method, vol), sub in pred.groupby(["method", "vol_bin"]):
        dice_row = stage4[stage4["metodo"].eq(method) & stage4["vol_bin"].eq(vol)]
        rows.append({
            "method": method,
            "vol_bin": vol,
            "n": int(len(sub)),
            "surface_hd95_mean": float(sub["surface_hd95"].mean()),
            "surface_hd95_median": float(sub["surface_hd95"].median()),
            "surface_asd_mean": float(sub["surface_asd"].mean()),
            "surface_asd_median": float(sub["surface_asd"].median()),
            "surface_chamfer_median": float(sub["surface_chamfer"].median()),
            "fallback_count": int(sub["pred_recon_fallback_used"].sum()),
            "lesionwise_dice_mean": float(dice_row["lesionwise_dice_mean"].iloc[0]) if len(dice_row) else np.nan,
            "global_dice_mean": float(dice_row["global_dice_mean"].iloc[0]) if len(dice_row) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values(["vol_bin", "surface_asd_median"])
    out.to_csv(OUT / "surface_prediction_fidelity_by_method_stratum.csv", index=False)
    return out


def dice_surface_relationship(pred_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vol in VOL_ORDER:
        sub = pred_summary[pred_summary["vol_bin"].eq(vol)].dropna(subset=["lesionwise_dice_mean", "surface_asd_median"])
        if len(sub) >= 3:
            rho_asd, p_asd = spearmanr(sub["lesionwise_dice_mean"], sub["surface_asd_median"])
            rho_hd, p_hd = spearmanr(sub["lesionwise_dice_mean"], sub["surface_hd95_median"])
        else:
            rho_asd = p_asd = rho_hd = p_hd = np.nan
        rows.append({
            "vol_bin": vol,
            "n_methods": int(len(sub)),
            "spearman_lesionwise_vs_surface_asd_median": float(rho_asd) if np.isfinite(rho_asd) else np.nan,
            "p_asd": float(p_asd) if np.isfinite(p_asd) else np.nan,
            "spearman_lesionwise_vs_surface_hd95_median": float(rho_hd) if np.isfinite(rho_hd) else np.nan,
            "p_hd95": float(p_hd) if np.isfinite(p_hd) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "surface_dice_relationship_by_stratum.csv", index=False)
    return out


def sample_for_plot(points: np.ndarray, max_points: int, seed: str) -> np.ndarray:
    return deterministic_sample(points, max_points, (seed, "plot"))


def render_overlay(case_id: str,
                   gt_points: np.ndarray,
                   pred_points_by_method: Dict[str, np.ndarray],
                   out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    views = [("axial x-y", 0, 1), ("coronal x-z", 0, 2), ("sagittal y-z", 1, 2)]
    colors = {"GT_ET": "lime", "gmm_T1c": "red", "variational_spline": "dodgerblue"}
    for ax, (title, i, j) in zip(axes, views):
        gt_plot = sample_for_plot(gt_points, MAX_RENDER_POINTS, case_id + "GT")
        if len(gt_plot):
            ax.scatter(gt_plot[:, i], gt_plot[:, j], s=0.4, c=colors["GT_ET"], alpha=0.45, label="GT ET")
        for method, pts in pred_points_by_method.items():
            pts_plot = sample_for_plot(pts, MAX_RENDER_POINTS, case_id + method)
            if len(pts_plot):
                ax.scatter(pts_plot[:, i], pts_plot[:, j], s=0.35, c=colors.get(method, "white"), alpha=0.28, label=method)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(case_id)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def irreducible_analysis() -> pd.DataFrame:
    rows = []
    for _, irr in pd.read_csv(IRREDUCIBLE).iterrows():
        case_id = str(irr["case_id"])
        seg, img = read_image(OLD20_DATA_ROOT / case_id / f"{case_id}-seg.nii.gz")
        gt = as_et(seg)
        gt_points, gt_meta = reconstruct_points(
            gt, img.GetSpacing(),
            SURF_DIR / "old20_irreducible" / case_id / "GT_ET_poisson_points.npz",
            (case_id, "old20", "GT_ET"),
        )
        pred_points_by_method = {}
        for method in ["gmm_T1c", "variational_spline"]:
            pred, _ = read_image(pred_path(case_id, method, OLD20_MASK_ROOT))
            pred = pred > 0
            pred_points, pred_meta = reconstruct_points(
                pred, img.GetSpacing(),
                SURF_DIR / "old20_irreducible" / case_id / f"{method}_poisson_points.npz",
                (case_id, "old20", method),
            )
            pred_points_by_method[method] = pred_points
            metrics = pointset_metrics(pred_points, gt_points)
            rows.append({
                "case_id": case_id,
                "method": method,
                "mechanism_class": irr["mechanism_class"],
                "gt_components": metric_component_count(gt),
                "pred_recon_status": pred_meta.get("status", ""),
                "surface_hd95_vs_gt_recon": metrics["hd95"],
                "surface_asd_vs_gt_recon": metrics["asd"],
                "surface_chamfer_vs_gt_recon": metrics["chamfer"],
                "interpretation": (
                    "cavity_or_vascular_confound_reconstructs_as_highly_separate_surface"
                    if metrics["asd"] > 20 else
                    "confound_surface_partly_tracks_gt_region"
                ),
            })
        render_overlay(case_id, gt_points, pred_points_by_method, FIG_DIR / f"{case_id}_irreducible_surface_overlay.png")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "surface_irreducible_cases.csv", index=False)
    return out


def situation_table(floor: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    merged = pred.merge(
        floor[["case_id", "gt_components", "thin_shell_proxy", "isoperimetric_quotient", "volume_surface_ratio"]],
        on="case_id",
        how="left",
    )
    rows = []
    situations = {
        "unifocal": merged["gt_components"].eq(1),
        "multifocal": merged["gt_components"].gt(1),
        "thin_shell_high_surface": merged["thin_shell_proxy"].ge(merged["thin_shell_proxy"].median()),
        "compact_low_surface": merged["thin_shell_proxy"].lt(merged["thin_shell_proxy"].median()),
        "low_isoperimetric_irregular": merged["isoperimetric_quotient"].le(merged["isoperimetric_quotient"].median()),
        "high_isoperimetric_compact": merged["isoperimetric_quotient"].gt(merged["isoperimetric_quotient"].median()),
    }
    for name, mask in situations.items():
        sub = merged[mask]
        rows.append({
            "situation": name,
            "n_case_method_rows": int(len(sub)),
            "surface_asd_median": float(sub["surface_asd"].median()),
            "surface_hd95_median": float(sub["surface_hd95"].median()),
            "surface_chamfer_median": float(sub["surface_chamfer"].median()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "surface_error_by_geometry_situation.csv", index=False)
    return out


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_empty_\n"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(c) for c in display.columns]
    rows = display.astype(str).values.tolist()
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def write_report(floor_by: pd.DataFrame,
                 corr: pd.DataFrame,
                 pred_by: pd.DataFrame,
                 dice_rel: pd.DataFrame,
                 irr: pd.DataFrame,
                 situations: pd.DataFrame) -> None:
    path = OUT / "SURFACE_RECONSTRUCTION_REPORT.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Surface-Based Evaluation and Reconstruction Analysis\n\n")
        f.write("Open3D Poisson reconstruction was run from ET boundary-voxel point clouds. ")
        f.write(f"Boundary clouds were capped at {MAX_BOUNDARY_POINTS} points and reconstructed at Poisson depth {POISSON_DEPTH}; ")
        f.write("surface metrics are symmetric point-set distances in millimeters: HD95, ASD, and squared-distance Chamfer.\n\n")
        f.write("## 1. GT Reconstruction Error Floor\n\n")
        f.write(df_to_markdown(floor_by))
        f.write("\n\n")
        top_corr = corr.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False).head(8)
        f.write("Strongest geometry/error associations:\n\n")
        f.write(df_to_markdown(top_corr))
        f.write("\n\n")
        f.write("## 2. Prediction Surface Fidelity\n\n")
        large = pred_by[pred_by["vol_bin"].eq("large")].sort_values("surface_asd_median")
        f.write("Large stratum, n=17:\n\n")
        f.write(df_to_markdown(large))
        f.write("\n\nAll feasible strata:\n\n")
        f.write(df_to_markdown(pred_by))
        f.write("\n\nDice-vs-surface relationship:\n\n")
        f.write(df_to_markdown(dice_rel))
        f.write("\n\n")
        f.write("## 3. Why Post-Treatment ET Is Geometrically Hard\n\n")
        f.write("Surface error by geometric situation:\n\n")
        f.write(df_to_markdown(situations))
        f.write("\n\nIrreducible old-20 confound cases:\n\n")
        f.write(df_to_markdown(irr))
        f.write("\n\nRenders are saved in `phase2/surface_reconstruction/figures/`.\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SURF_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    process_present = manifest[manifest["process"].astype(int).eq(1) & manifest["vol_bin"].isin(VOL_ORDER)].copy()

    floor = reconstruction_floor(process_present)
    floor_by = floor_summary(floor)
    corr = floor_correlations(floor)
    pred = prediction_surface_fidelity(process_present)
    pred_by = prediction_summary(pred)
    dice_rel = dice_surface_relationship(pred_by)
    irr = irreducible_analysis()
    situations = situation_table(floor, pred)
    write_report(floor_by, corr, pred_by, dice_rel, irr, situations)

    print("\nGT RECONSTRUCTION FLOOR")
    print(floor_by.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nLARGE-STRATUM SURFACE FIDELITY")
    cols = ["method", "surface_asd_median", "surface_hd95_median", "lesionwise_dice_mean", "global_dice_mean"]
    print(pred_by[pred_by["vol_bin"].eq("large")][cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nDICE VS SURFACE")
    print(dice_rel.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nReport: {OUT / 'SURFACE_RECONSTRUCTION_REPORT.md'}")


if __name__ == "__main__":
    main()
