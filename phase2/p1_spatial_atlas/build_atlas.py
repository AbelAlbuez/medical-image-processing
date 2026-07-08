"""Build leave-fold spatial ET occurrence atlases and score atlas priors.

Inputs: the 100-case cohort manifest, cleaned cohort volumes, baseline masks,
and the locked Phase 2 metric in ``phase2.metrics``.
Outputs: atlas NIfTI files plus P1 sweep/result CSVs in this directory.
"""

from __future__ import annotations

from pathlib import Path
import ast
import json
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2.metrics import DEFAULT_CONFIG, fp_aware_lesionwise_dice


RUN_ROOT = ROOT / "analysis" / "pre_cohort_run" / "cohort_100_core"
CLEAN_ROOT = RUN_ROOT / "limpieza"
SEG_ROOT = RUN_ROOT / "segmentacion"
MANIFEST = ROOT / "cohort" / "COHORT_MANIFEST_selected.csv"
STAGE4_CASE_METRICS = ROOT / "analysis" / "stage4_metrics" / "stage4_case_metrics.csv"
BASELINE_TARGETS = ROOT / "phase2" / "baseline_targets_per_axis.csv"
IRREDUCIBLE = ROOT / "analysis" / "stage3d_irreducible_mechanism.csv"
OUT = ROOT / "phase2" / "p1_spatial_atlas"
ATLAS_DIR = OUT / "atlases"
FIG_DIR = OUT / "figures"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
VOL_ORDER = ["small", "medium", "large"]
PROB_THRESHOLDS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
MAP_THRESHOLDS = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
ATLAS_SIGMA_VOX = 2.0
BOOTSTRAPS = 2000
RNG_SEED = 0


def load_image(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def as_et(seg_arr: np.ndarray) -> np.ndarray:
    return np.round(seg_arr).astype(np.int16) == 3


def save_like(arr_zyx: np.ndarray, ref_img: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    img.CopyInformation(ref_img)
    sitk.WriteImage(img, str(path), useCompression=True)


def verify_grid(process: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref = None
    for case_id in process["case_id"].astype(str):
        for source, path in [
            ("clean_t1c", CLEAN_ROOT / case_id / f"{case_id}-t1c.nii.gz"),
            ("clean_seg", CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz"),
        ]:
            img = sitk.ReadImage(str(path))
            record = {
                "case_id": case_id,
                "source": source,
                "size_xyz": "x".join(map(str, img.GetSize())),
                "spacing": tuple(float(x) for x in img.GetSpacing()),
                "origin": tuple(float(x) for x in img.GetOrigin()),
                "direction": tuple(float(x) for x in img.GetDirection()),
            }
            if ref is None and source == "clean_t1c":
                ref = record
            record["matches_reference"] = (
                record["size_xyz"] == ref["size_xyz"]
                and np.allclose(record["spacing"], ref["spacing"], atol=1e-6)
                and np.allclose(record["origin"], ref["origin"], atol=1e-6)
                and np.allclose(record["direction"], ref["direction"], atol=1e-6)
            )
            rows.append(record)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p1_grid_check.csv", index=False)
    return out


def build_atlases(process: pd.DataFrame) -> Dict[int, np.ndarray]:
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    atlases: Dict[int, np.ndarray] = {}
    ref_img = sitk.ReadImage(str(CLEAN_ROOT / process.iloc[0]["case_id"] / f"{process.iloc[0]['case_id']}-t1c.nii.gz"))
    for fold in sorted(process["fold"].dropna().astype(int).unique()):
        atlas_path = ATLAS_DIR / f"et_occurrence_atlas_holdout_fold{fold}.nii.gz"
        if atlas_path.exists():
            atlases[int(fold)] = sitk.GetArrayFromImage(sitk.ReadImage(str(atlas_path))).astype(np.float32)
            continue
        train = process[
            (process["fold"].astype(int) != fold)
            & (process["vol_bin"] != "absent")
        ].copy()
        acc = None
        for case_id in train["case_id"].astype(str):
            seg_arr, _ = load_image(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
            et = as_et(seg_arr).astype(np.float32)
            if acc is None:
                acc = np.zeros_like(et, dtype=np.float32)
            acc += et
        atlas = acc / max(1, len(train))
        atlas = ndimage.gaussian_filter(atlas, sigma=ATLAS_SIGMA_VOX).astype(np.float32)
        atlas = np.clip(atlas, 0.0, 1.0)
        atlases[int(fold)] = atlas
        save_like(atlas, ref_img, atlas_path)
    return atlases


def render_atlas_sanity(atlases: Dict[int, np.ndarray]) -> pd.DataFrame:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mean_atlas = np.mean(np.stack(list(atlases.values()), axis=0), axis=0)
    z, y, x = [s // 2 for s in mean_atlas.shape]
    planes = [
        ("axial_z_mid", mean_atlas[z, :, :]),
        ("coronal_y_mid", mean_atlas[:, y, :]),
        ("sagittal_x_mid", mean_atlas[:, :, x]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (title, plane) in zip(axes, planes):
        im = ax.imshow(np.flipud(plane), cmap="magma", vmin=0, vmax=max(0.001, float(mean_atlas.max())))
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle("Mean leave-fold ET-occurrence atlas")
    fig.savefig(FIG_DIR / "atlas_mean_midplanes.png", dpi=180)
    plt.close(fig)

    rows = []
    for fold, atlas in sorted(atlases.items()):
        nz = atlas[atlas > 0]
        rows.append({
            "fold": fold,
            "min": float(atlas.min()),
            "max": float(atlas.max()),
            "mean": float(atlas.mean()),
            "nonzero_voxels": int((atlas > 0).sum()),
            "p50_nonzero": float(np.percentile(nz, 50)) if nz.size else 0.0,
            "p95_nonzero": float(np.percentile(nz, 95)) if nz.size else 0.0,
            "p99_nonzero": float(np.percentile(nz, 99)) if nz.size else 0.0,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p1_atlas_summary_stats.csv", index=False)
    return out


def parse_centroid(text: str) -> Tuple[int, int, int]:
    vals = [float(x) for x in str(text).split(";")]
    return tuple(int(round(v)) for v in vals)


def smoking_gun_values(atlases: Dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    irr = pd.read_csv(IRREDUCIBLE)
    for _, row in irr.iterrows():
        z, y, x = parse_centroid(row["gmm_centroid_zyx"])
        values = {f"fold{fold}_atlas_value": float(atlas[z, y, x])
                  for fold, atlas in sorted(atlases.items())}
        vals = list(values.values())
        rows.append({
            "case_id": row["case_id"],
            "mechanism_class": row["mechanism_class"],
            "gmm_centroid_zyx": row["gmm_centroid_zyx"],
            "atlas_value_mean": float(np.mean(vals)),
            "atlas_value_min": float(np.min(vals)),
            "atlas_value_max": float(np.max(vals)),
            **values,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p1_smoking_gun_seed_atlas_values.csv", index=False)
    return out


def post_filter(mask: np.ndarray, atlas: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return mask.astype(bool)
    labels, n = ndimage.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return mask.astype(bool)
    keep = np.zeros(n + 1, dtype=bool)
    sums = ndimage.sum(atlas, labels, index=np.arange(1, n + 1))
    sizes = ndimage.sum(mask.astype(np.uint8), labels, index=np.arange(1, n + 1))
    means = np.divide(sums, sizes, out=np.zeros_like(sums, dtype=float), where=sizes > 0)
    keep[1:] = means >= threshold
    return keep[labels]


def enhancement_score(case_id: str) -> np.ndarray:
    t1c, _ = load_image(CLEAN_ROOT / case_id / f"{case_id}-t1c.nii.gz")
    t1n, _ = load_image(CLEAN_ROOT / case_id / f"{case_id}-t1n.nii.gz")
    diff = np.maximum(t1c.astype(np.float32) - t1n.astype(np.float32), 0.0)
    positive = diff[diff > 0]
    if positive.size == 0:
        return np.zeros_like(diff, dtype=np.float32)
    scale = float(np.percentile(positive, 99.5))
    if scale <= 0:
        return np.zeros_like(diff, dtype=np.float32)
    return np.clip(diff / scale, 0.0, 1.0).astype(np.float32)


def score_fast(gt: np.ndarray, pred: np.ndarray, vol_bin: str) -> Dict[str, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_vox = int(gt.sum())
    pred_vox = int(pred.sum())
    if gt_vox == 0:
        return {
            "vol_bin": vol_bin,
            "gt_vox": gt_vox,
            "pred_vox": pred_vox,
            "lesionwise_dice_mean": np.nan,
            "lesion_tp": np.nan,
            "lesion_fn": np.nan,
            "lesion_fp": np.nan,
            "flood_gt_10000_vox": bool(pred_vox > DEFAULT_CONFIG.flood_threshold_vox),
            "correct_absent_pred_lt_10_vox": bool(
                pred_vox < DEFAULT_CONFIG.absent_tolerance_vox),
        }

    lesion = fp_aware_lesionwise_dice(gt, pred, DEFAULT_CONFIG.min_component_size_vox)
    return {
        "vol_bin": vol_bin,
        "gt_vox": gt_vox,
        "pred_vox": pred_vox,
        "lesionwise_dice_mean": lesion["lesionwise_dice_mean"],
        "lesion_tp": lesion["lesion_tp"],
        "lesion_fn": lesion["lesion_fn"],
        "lesion_fp": lesion["lesion_fp"],
        "flood_gt_10000_vox": bool(pred_vox > DEFAULT_CONFIG.flood_threshold_vox),
        "correct_absent_pred_lt_10_vox": (
            bool(pred_vox < DEFAULT_CONFIG.absent_tolerance_vox) if gt_vox == 0 else np.nan
        ),
    }


def evaluate_sweeps(process: pd.DataFrame, atlases: Dict[int, np.ndarray]) -> pd.DataFrame:
    case_dir = OUT / "case_sweep_metrics"
    case_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    for idx, meta in process.reset_index(drop=True).iterrows():
        case_id = str(meta["case_id"])
        case_csv = case_dir / f"{case_id}.csv"
        if case_csv.exists():
            print(f"[P1 eval] {idx+1:03d}/{len(process)} {case_id} [cache]", flush=True)
            frames.append(pd.read_csv(case_csv))
            continue
        rows: List[Dict[str, object]] = []
        fold = int(meta["fold"])
        vol_bin = str(meta["vol_bin"])
        atlas = atlases[fold]
        gt_arr, _ = load_image(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
        gt = as_et(gt_arr)
        print(f"[P1 eval] {idx+1:03d}/{len(process)} {case_id}", flush=True)

        baseline_masks = {}
        for method in METHODS:
            pred, _ = load_image(SEG_ROOT / case_id / f"{case_id}-et_{method}.nii.gz")
            baseline_masks[method] = pred > 0

        for t in PROB_THRESHOLDS:
            atlas_gate = atlas >= t
            for method, pred in baseline_masks.items():
                for mode, pred2 in [
                    ("pre_filter", pred & atlas_gate),
                    ("post_filter", post_filter(pred, atlas, t)),
                ]:
                    rows.append({
                        "case_id": case_id,
                        "fold": fold,
                        "mode": mode,
                        "method": method,
                        "threshold": t,
                        **score_fast(gt, pred2, vol_bin),
                    })

        enh = enhancement_score(case_id)
        map_score = atlas * enh
        for t in MAP_THRESHOLDS:
            pred2 = map_score >= t
            rows.append({
                "case_id": case_id,
                "fold": fold,
                "mode": "map_atlas_x_enhancement",
                "method": "atlas_map",
                "threshold": t,
                **score_fast(gt, pred2, vol_bin),
            })
        case_df = pd.DataFrame(rows)
        case_df.to_csv(case_csv, index=False)
        frames.append(case_df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "p1_spatial_sweep_case_metrics.csv", index=False)
    return out


def aggregate_configs(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, method, threshold), sub in case_metrics.groupby(["mode", "method", "threshold"]):
        absent = sub[sub["vol_bin"].eq("absent")]
        row = {
            "mode": mode,
            "method": method,
            "threshold": threshold,
            "n_cases": sub["case_id"].nunique(),
            "absent_n": len(absent),
            "absent_flood_rate": float(absent["flood_gt_10000_vox"].mean()) if len(absent) else np.nan,
            "absent_median_fp_vox": float(absent["pred_vox"].median()) if len(absent) else np.nan,
            "absent_max_fp_vox": int(absent["pred_vox"].max()) if len(absent) else 0,
        }
        for vol in VOL_ORDER:
            s = sub[sub["vol_bin"].eq(vol)]
            row[f"{vol}_n"] = len(s)
            row[f"{vol}_lesionwise_mean"] = float(s["lesionwise_dice_mean"].mean())
            row[f"{vol}_lesion_tp"] = int(s["lesion_tp"].sum(skipna=True))
            row[f"{vol}_lesion_fn"] = int(s["lesion_fn"].sum(skipna=True))
            row[f"{vol}_lesion_fp"] = int(s["lesion_fp"].sum(skipna=True))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p1_spatial_sweep_summary.csv", index=False)
    return out


def detection_baseline_by_case() -> pd.DataFrame:
    stage4 = pd.read_csv(STAGE4_CASE_METRICS)
    targets = pd.read_csv(BASELINE_TARGETS)
    det_best = targets[(targets["axis"].eq("detection")) & (targets["is_axis_best"])].copy()
    rows = []
    for _, row in det_best.iterrows():
        vol = str(row["target_scope"]).replace("present_", "")
        method = row["metodo"]
        sub = stage4[(stage4["vol_bin"].eq(vol)) & (stage4["metodo"].eq(method))]
        rows.append(sub[["case_id", "vol_bin", "lesionwise_dice_mean"]].assign(
            detection_baseline_method=method,
            detection_baseline_lesionwise=sub["lesionwise_dice_mean"],
        ))
    return pd.concat(rows, ignore_index=True)[[
        "case_id", "vol_bin", "detection_baseline_method", "detection_baseline_lesionwise"
    ]]


def bootstrap_ci(values: np.ndarray, reducer=np.mean, n: int = BOOTSTRAPS) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(RNG_SEED)
    stats = []
    for _ in range(n):
        sample = values[rng.integers(0, values.size, size=values.size)]
        stats.append(float(reducer(sample)))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def key_table(sweep_cases: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    stage4 = pd.read_csv(STAGE4_CASE_METRICS)
    otsu_abs = stage4[(stage4["metodo"].eq("otsu_T1c")) & (stage4["vol_bin"].eq("absent"))]
    otsu_abs = otsu_abs[["case_id", "pred_vox", "flood_gt_10000_vox"]].rename(
        columns={"pred_vox": "otsu_pred_vox", "flood_gt_10000_vox": "otsu_flood"})
    det_base = detection_baseline_by_case()

    candidates = []
    for mode, sub in summary.groupby("mode"):
        ranked = sub.sort_values(
            ["absent_flood_rate", "absent_median_fp_vox",
             "large_lesionwise_mean", "medium_lesionwise_mean", "small_lesionwise_mean"],
            ascending=[True, True, False, False, False])
        candidates.append(ranked.iloc[0].to_dict() | {"selection": "best_fp"})
        viable = sub[
            (sub["small_lesionwise_mean"] >= float(
                det_base.merge(sweep_cases[["case_id", "vol_bin"]].drop_duplicates(), on=["case_id", "vol_bin"])
                [lambda d: d["vol_bin"].eq("small")]["detection_baseline_lesionwise"].mean()) - 1e-12)
            & (sub["medium_lesionwise_mean"] >= float(
                det_base.merge(sweep_cases[["case_id", "vol_bin"]].drop_duplicates(), on=["case_id", "vol_bin"])
                [lambda d: d["vol_bin"].eq("medium")]["detection_baseline_lesionwise"].mean()) - 1e-12)
            & (sub["large_lesionwise_mean"] >= float(
                det_base.merge(sweep_cases[["case_id", "vol_bin"]].drop_duplicates(), on=["case_id", "vol_bin"])
                [lambda d: d["vol_bin"].eq("large")]["detection_baseline_lesionwise"].mean()) - 1e-12)
        ]
        if not viable.empty:
            candidates.append(viable.sort_values(
                ["absent_flood_rate", "absent_median_fp_vox"],
                ascending=[True, True]).iloc[0].to_dict() | {"selection": "best_no_detection_point_regression"})

    rows = []
    for cand in candidates:
        sub = sweep_cases[
            sweep_cases["mode"].eq(cand["mode"])
            & sweep_cases["method"].eq(cand["method"])
            & np.isclose(sweep_cases["threshold"], cand["threshold"])
        ].copy()
        absent = sub[sub["vol_bin"].eq("absent")].merge(otsu_abs, on="case_id", how="inner")
        flood_delta = absent["flood_gt_10000_vox"].astype(float) - absent["otsu_flood"].astype(float)
        median_delta_samples = absent["pred_vox"].to_numpy(float) - absent["otsu_pred_vox"].to_numpy(float)
        flood_ci = bootstrap_ci(flood_delta.to_numpy(float), np.mean)
        median_ci = bootstrap_ci(median_delta_samples, np.median)

        row = dict(cand)
        row["delta_absent_flood_vs_otsu"] = float(flood_delta.mean())
        row["delta_absent_flood_ci_low"] = flood_ci[0]
        row["delta_absent_flood_ci_high"] = flood_ci[1]
        row["delta_absent_median_fp_vs_otsu"] = float(
            absent["pred_vox"].median() - absent["otsu_pred_vox"].median())
        row["paired_absent_fp_vox_delta_median_ci_low"] = median_ci[0]
        row["paired_absent_fp_vox_delta_median_ci_high"] = median_ci[1]

        present = sub[sub["vol_bin"].isin(VOL_ORDER)].merge(det_base, on=["case_id", "vol_bin"], how="inner")
        for vol in VOL_ORDER:
            pv = present[present["vol_bin"].eq(vol)].copy()
            delta = (pv["lesionwise_dice_mean"] - pv["detection_baseline_lesionwise"]).to_numpy(float)
            ci = bootstrap_ci(delta, np.mean)
            row[f"{vol}_delta_vs_detection_best"] = float(np.nanmean(delta)) if delta.size else np.nan
            row[f"{vol}_delta_ci_low"] = ci[0]
            row[f"{vol}_delta_ci_high"] = ci[1]
            row[f"{vol}_detection_baseline_method"] = (
                pv["detection_baseline_method"].iloc[0] if len(pv) else ""
            )
        row["no_point_detection_regression"] = all(
            row[f"{vol}_delta_vs_detection_best"] >= -1e-12 for vol in VOL_ORDER
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p1_key_comparison_table.csv", index=False)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    process = pd.read_csv(MANIFEST)
    process = process[process["process"].astype(int).eq(1)].copy()

    grid = verify_grid(process)
    if not bool(grid["matches_reference"].all()):
        print(grid[~grid["matches_reference"]].to_string(index=False))
        raise SystemExit("Grid check failed; atlas requires common space.")

    atlases = build_atlases(process)
    atlas_stats = render_atlas_sanity(atlases)
    smoke = smoking_gun_values(atlases)

    sweep_cases = evaluate_sweeps(process, atlases)
    summary = aggregate_configs(sweep_cases)
    key = key_table(sweep_cases, summary)

    run_summary = {
        "grid_passed": bool(grid["matches_reference"].all()),
        "atlas_sigma_vox": ATLAS_SIGMA_VOX,
        "prob_thresholds": PROB_THRESHOLDS,
        "map_thresholds": MAP_THRESHOLDS,
        "smoking_gun_max_value": float(smoke["atlas_value_max"].max()),
        "figure": str(FIG_DIR / "atlas_mean_midplanes.png"),
        "comparison": (
            "Each held-out case uses the atlas built from the other four folds only. "
            "Spatial configs are compared to otsu_T1c on absent FP burden and to the "
            "per-stratum detection-best baseline on present lesion-wise Dice."
        ),
    }
    (OUT / "p1_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("\nGRID")
    print(grid.groupby("source")["matches_reference"].agg(["count", "sum"]).to_string())
    print("\nATLAS STATS")
    print(atlas_stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSMOKING GUN")
    print(smoke.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print("\nKEY TABLE")
    print(key.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
