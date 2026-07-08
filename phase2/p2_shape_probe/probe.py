"""Probe whether predicted false-positive components are shape-separable.

Inputs are existing baseline masks, cohort masks, and GT components. Outputs are
component-level shape proxy tables and separation summaries for deciding whether
the shape prior is worth evaluating.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "BraTS2024-BraTS-GLI-TrainingData" / "training_data1_v2"
COHORT = ROOT / "cohort" / "COHORT_MANIFEST_selected.csv"
MASK_ROOT = ROOT / "analysis" / "pre_cohort_run" / "cohort_100_core" / "segmentacion"
OLD20_MASK_ROOT = ROOT / "analysis" / "stage5_runner_validation" / "run_20260706_220739" / "segmentacion"
OUT_DIR = ROOT / "phase2" / "p2_shape_probe"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
MIN_COMPONENT_SIZE = 10
FLOOD_VOX_THRESHOLD = 10_000
PERI_CAVITY_CASES = ["BraTS-GLI-00533-100", "BraTS-GLI-02078-100"]
PERI_CAVITY_METHODS = ["gmm_T1c", "variational_spline"]
MAX_FP_COMPONENTS_PER_MASK = 1


def read_mask(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def seg_path(case_id: str) -> Path:
    return DATA_ROOT / case_id / f"{case_id}-seg.nii.gz"


def pred_path(mask_root: Path, case_id: str, method: str) -> Path:
    return mask_root / case_id / f"{case_id}-et_{method}.nii.gz"


def bbox_extent(slc: tuple[slice, ...]) -> tuple[int, int, int]:
    return tuple(int(s.stop - s.start) for s in slc)


def surface_area_voxels(component: np.ndarray) -> float:
    if component.sum() == 0:
        return float("nan")
    padded = np.pad(component.astype(np.uint8), 1, mode="constant")
    # Fast voxel-face surface proxy: count exposed unit faces on the 6-neighborhood.
    # This is blockier than marching cubes, but stable and appropriate for a shape screen.
    dz = np.abs(np.diff(padded, axis=0)).sum()
    dy = np.abs(np.diff(padded, axis=1)).sum()
    dx = np.abs(np.diff(padded, axis=2)).sum()
    return float(dz + dy + dx)


def inscribed_radius_adaptive(component: np.ndarray) -> tuple[float, int]:
    voxels = int(component.size)
    if voxels <= 350_000:
        return float(ndi.distance_transform_edt(component).max()), 1
    step = 2 if voxels <= 2_000_000 else 4
    coarse = component[::step, ::step, ::step]
    return float(ndi.distance_transform_edt(coarse).max() * step), step


def component_rows(
    mask: np.ndarray,
    case_id: str,
    source_group: str,
    method: str,
    gt: np.ndarray | None = None,
    max_components: int | None = None,
) -> list[dict]:
    structure = ndi.generate_binary_structure(3, 3)
    labels, n_labels = ndi.label(mask.astype(bool), structure=structure)
    if n_labels == 0:
        return []
    counts = np.bincount(labels.ravel())
    valid_labels = np.flatnonzero(counts >= MIN_COMPONENT_SIZE)
    valid_labels = valid_labels[valid_labels != 0]
    if len(valid_labels) == 0:
        return []
    if max_components is not None and len(valid_labels) > max_components:
        order = np.argsort(counts[valid_labels])[::-1]
        valid_labels = valid_labels[order[:max_components]]
    valid_label_set = set(int(label) for label in valid_labels)

    objects = ndi.find_objects(labels)
    rows: list[dict] = []
    for label_idx, slc in enumerate(objects, start=1):
        if label_idx not in valid_label_set:
            continue
        if slc is None:
            continue
        volume = int(counts[label_idx])
        crop_labels = labels[slc]
        component = crop_labels == label_idx

        extent_z, extent_y, extent_x = bbox_extent(slc)
        bbox_diag = float(math.sqrt(extent_z**2 + extent_y**2 + extent_x**2))
        max_extent = float(max(extent_z, extent_y, extent_x))
        min_extent = float(max(1, min(extent_z, extent_y, extent_x)))
        elongation = max_extent / min_extent

        surface = surface_area_voxels(component)
        compactness = volume / (surface**1.5) if surface and surface > 0 else float("nan")
        isoperimetric = (36.0 * math.pi * (volume**2) / (surface**3)) if surface and surface > 0 else float("nan")

        inscribed_radius, inscribed_radius_step = inscribed_radius_adaptive(component)
        radius_over_bbox_diag = inscribed_radius / bbox_diag if bbox_diag > 0 else float("nan")
        equivalent_sphere_radius = ((3.0 * volume) / (4.0 * math.pi)) ** (1.0 / 3.0)
        radius_over_equiv_sphere = inscribed_radius / equivalent_sphere_radius if equivalent_sphere_radius > 0 else float("nan")

        eroded = ndi.binary_erosion(component, structure=np.ones((3, 3, 3)), iterations=1)
        eroded_volume = int(eroded.sum())
        _, eroded_components = ndi.label(eroded, structure=structure)
        erosion_survival_frac = eroded_volume / volume if volume else 0.0

        overlap_gt_vox = np.nan
        if gt is not None:
            overlap_gt_vox = int((component & (gt[slc] > 0)).sum())

        rows.append(
            {
                "source_group": source_group,
                "case_id": case_id,
                "method": method,
                "component_label": label_idx,
                "volume_vox": volume,
                "surface_area_vox": surface,
                "compactness_v_over_s15": compactness,
                "isoperimetric_quotient": isoperimetric,
                "inscribed_radius_vox": inscribed_radius,
                "inscribed_radius_grid_step": inscribed_radius_step,
                "bbox_diag_vox": bbox_diag,
                "max_extent_vox": max_extent,
                "elongation_max_over_min": elongation,
                "radius_over_bbox_diag": radius_over_bbox_diag,
                "radius_over_equiv_sphere": radius_over_equiv_sphere,
                "eroded_subcomponents": int(eroded_components),
                "erosion_survival_frac": erosion_survival_frac,
                "overlap_gt_vox": overlap_gt_vox,
                "component_scope": "all_ge10" if max_components is None else f"top_{max_components}_largest_ge10",
                "bbox_z0": slc[0].start,
                "bbox_z1": slc[0].stop,
                "bbox_y0": slc[1].start,
                "bbox_y1": slc[1].stop,
                "bbox_x0": slc[2].start,
                "bbox_x1": slc[2].stop,
            }
        )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "volume_vox",
        "surface_area_vox",
        "compactness_v_over_s15",
        "isoperimetric_quotient",
        "inscribed_radius_vox",
        "radius_over_bbox_diag",
        "radius_over_equiv_sphere",
        "elongation_max_over_min",
        "eroded_subcomponents",
        "erosion_survival_frac",
    ]
    rows = []
    for group, sub in df.groupby("source_group", dropna=False):
        row = {"source_group": group, "n_components": len(sub), "n_cases": sub["case_id"].nunique()}
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q25"] = values.quantile(0.25)
            row[f"{metric}_q75"] = values.quantile(0.75)
            row[f"{metric}_mean"] = values.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def effect_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    true = df[df["source_group"] == "true_et_gt"]
    metrics = [
        "compactness_v_over_s15",
        "isoperimetric_quotient",
        "inscribed_radius_vox",
        "radius_over_bbox_diag",
        "radius_over_equiv_sphere",
        "elongation_max_over_min",
        "erosion_survival_frac",
    ]
    for false_group in ["absent_fp", "peri_cavity_seed_fp"]:
        false = df[df["source_group"] == false_group]
        for metric in metrics:
            a = pd.to_numeric(true[metric], errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(false[metric], errors="coerce").dropna().to_numpy()
            if len(a) == 0 or len(b) == 0:
                continue
            alternative = "greater"
            stat = mannwhitneyu(a, b, alternative=alternative)
            auc_true_greater = float(stat.statistic / (len(a) * len(b)))
            rows.append(
                {
                    "comparison": f"true_et_gt_vs_{false_group}",
                    "metric": metric,
                    "true_median": float(np.median(a)),
                    "false_median": float(np.median(b)),
                    "true_q25": float(np.quantile(a, 0.25)),
                    "true_q75": float(np.quantile(a, 0.75)),
                    "false_q25": float(np.quantile(b, 0.25)),
                    "false_q75": float(np.quantile(b, 0.75)),
                    "mann_whitney_alt_true_greater_p": float(stat.pvalue),
                    "auc_probability_true_greater": auc_true_greater,
                    "n_true": int(len(a)),
                    "n_false": int(len(b)),
                }
            )
    return pd.DataFrame(rows)


def write_boxplot(df: pd.DataFrame) -> None:
    metrics = [
        ("isoperimetric_quotient", "Isoperimetric quotient"),
        ("radius_over_bbox_diag", "Inscribed radius / bbox diagonal"),
        ("erosion_survival_frac", "1-voxel erosion survival"),
        ("elongation_max_over_min", "BBox elongation"),
    ]
    groups = ["true_et_gt", "absent_fp", "peri_cavity_seed_fp"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        data = [pd.to_numeric(df[df["source_group"] == group][metric], errors="coerce").dropna() for group in groups]
        ax.boxplot(data, labels=["true ET", "absent FP", "peri-cavity FP"], showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=20)
    fig.savefig(OUT_DIR / "p2_shape_probe_boxplots.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(COHORT)
    process = manifest[manifest["process"] == 1].copy()
    present_cases = process[process["et_present"] == 1]["case_id"].tolist()
    absent_cases = process[process["et_present"] == 0]["case_id"].tolist()

    rows: list[dict] = []

    for case_id in present_cases:
        seg = read_mask(seg_path(case_id))
        rows.extend(component_rows(seg == 3, case_id, "true_et_gt", "GT_ET", None))

    absent_case_rows = []
    for case_id in absent_cases:
        for method in METHODS:
            path = pred_path(MASK_ROOT, case_id, method)
            pred = read_mask(path) > 0
            pred_vox = int(pred.sum())
            if pred_vox <= FLOOD_VOX_THRESHOLD:
                continue
            comp = component_rows(pred, case_id, "absent_fp", method, None, max_components=MAX_FP_COMPONENTS_PER_MASK)
            rows.extend(comp)
            absent_case_rows.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "pred_vox": pred_vox,
                    "n_components_ge10": len(comp),
                }
            )

    peri_rows = []
    for case_id in PERI_CAVITY_CASES:
        gt = read_mask(seg_path(case_id)) == 3
        for method in PERI_CAVITY_METHODS:
            path = pred_path(OLD20_MASK_ROOT, case_id, method)
            pred = read_mask(path) > 0
            comp = component_rows(pred, case_id, "peri_cavity_seed_fp", method, gt, max_components=MAX_FP_COMPONENTS_PER_MASK)
            fp_comp = [row for row in comp if row["overlap_gt_vox"] == 0]
            rows.extend(fp_comp)
            peri_rows.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "pred_vox": int(pred.sum()),
                    "gt_et_vox": int(gt.sum()),
                    "n_components_ge10": len(comp),
                    "n_fp_components_ge10": len(fp_comp),
                    "fp_component_vox_total": int(sum(row["volume_vox"] for row in fp_comp)),
                    "overlap_gt_vox_total": int((pred & gt).sum()),
                }
            )

    component_df = pd.DataFrame(rows)
    component_df.to_csv(OUT_DIR / "p2_shape_components.csv", index=False)

    summary_df = summarize(component_df)
    summary_df.to_csv(OUT_DIR / "p2_shape_distribution_summary.csv", index=False)

    effects_df = effect_tests(component_df)
    effects_df.to_csv(OUT_DIR / "p2_shape_separation_tests.csv", index=False)

    pd.DataFrame(absent_case_rows).to_csv(OUT_DIR / "p2_absent_flood_case_inventory.csv", index=False)
    pd.DataFrame(peri_rows).to_csv(OUT_DIR / "p2_peri_cavity_seed_inventory.csv", index=False)
    write_boxplot(component_df)

    print("Wrote:")
    for name in [
        "p2_shape_components.csv",
        "p2_shape_distribution_summary.csv",
        "p2_shape_separation_tests.csv",
        "p2_absent_flood_case_inventory.csv",
        "p2_peri_cavity_seed_inventory.csv",
        "p2_shape_probe_boxplots.png",
    ]:
        print(f"  {OUT_DIR / name}")
    print()
    print(summary_df.to_string(index=False))
    print()
    print(effects_df.to_string(index=False))


if __name__ == "__main__":
    main()
