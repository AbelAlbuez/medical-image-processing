from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "analysis" / "pre_cohort_run" / "cohort_100_core"
SEG_ROOT = RUN_ROOT / "segmentacion"
CLEAN_ROOT = RUN_ROOT / "limpieza"
MANIFEST = ROOT / "cohort" / "COHORT_MANIFEST_selected.csv"
BASELINE = ROOT / "analysis" / "baseline" / "metricas_ET_baseline.csv"
OUT = ROOT / "analysis" / "stage4_metrics"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
VOL_ORDER = ["absent", "small", "medium", "large"]
ABSENT_TOLERANCE_VOX = 10
FLOOD_THRESHOLD_VOX = 10_000
LARGE_FP_THRESHOLD_VOX = 1_000
MIN_COMPONENT_SIZE_VOX = 10
STRUCT26 = np.ones((3, 3, 3), dtype=bool)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def load_mask(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def component_labels(mask: np.ndarray,
                     min_size: int = MIN_COMPONENT_SIZE_VOX) -> Tuple[np.ndarray, int, np.ndarray]:
    labels, n = ndimage.label(mask.astype(bool), structure=STRUCT26)
    if n == 0:
        return labels, 0, np.zeros(1, dtype=np.int64)
    sizes = np.bincount(labels.ravel())
    if min_size > 1:
        keep = [label for label in range(1, n + 1) if sizes[label] >= min_size]
        labels, n = ndimage.label(np.isin(labels, keep), structure=STRUCT26)
        if n == 0:
            return labels, 0, np.zeros(1, dtype=np.int64)
    sizes = np.bincount(labels.ravel())
    return labels, n, sizes


def lesionwise_dice(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt_lab, n_gt, gt_sizes = component_labels(gt)
    pred_lab, n_pred, pred_sizes = component_labels(pred)

    out: Dict[str, float] = {
        "gt_components": int(n_gt),
        "pred_components": int(n_pred),
        "matched_components": 0,
        "lesion_tp": 0,
        "lesion_fn": int(n_gt),
        "lesion_fp": int(n_pred),
        "lesion_dice_sum": 0.0,
        "lesionwise_dice_mean": np.nan,
        "lesionwise_dice_median": np.nan,
        "lesion_detection_rate_dice_gt_0": np.nan,
        "lesion_detection_rate_dice_ge_0_1": np.nan,
    }
    if n_gt == 0:
        return out
    if n_pred == 0:
        out.update({
            "matched_components": 0,
            "lesion_tp": 0,
            "lesion_fn": int(n_gt),
            "lesion_fp": 0,
            "lesion_dice_sum": 0.0,
            "lesionwise_dice_mean": 0.0,
            "lesionwise_dice_median": 0.0,
            "lesion_detection_rate_dice_gt_0": 0.0,
            "lesion_detection_rate_dice_ge_0_1": 0.0,
        })
        return out

    both = (gt_lab > 0) & (pred_lab > 0)
    scores = np.zeros((n_gt, n_pred), dtype=np.float32)
    if np.any(both):
        pairs = np.stack([gt_lab[both], pred_lab[both]], axis=1)
        pair_ids, counts = np.unique(pairs, axis=0, return_counts=True)
        for (g, p), inter in zip(pair_ids, counts):
            denom = gt_sizes[g] + pred_sizes[p]
            if denom > 0:
                scores[g - 1, p - 1] = 2.0 * inter / denom

    if scores.size == 0:
        per_gt = np.zeros(n_gt, dtype=np.float32)
        positive_matches = np.zeros(0, dtype=np.float32)
    else:
        gt_idx, pred_idx = linear_sum_assignment(-scores)
        per_gt = np.zeros(n_gt, dtype=np.float32)
        per_gt[gt_idx] = scores[gt_idx, pred_idx]
        positive_matches = scores[gt_idx, pred_idx][scores[gt_idx, pred_idx] > 0]

    tp = int(positive_matches.size)
    fn = int(n_gt - tp)
    fp = int(n_pred - tp)
    denom = tp + fn + fp
    lesionwise_fp_aware = float(positive_matches.sum() / denom) if denom else np.nan

    out.update({
        "matched_components": tp,
        "lesion_tp": tp,
        "lesion_fn": fn,
        "lesion_fp": fp,
        "lesion_dice_sum": float(positive_matches.sum()),
        "lesionwise_dice_mean": lesionwise_fp_aware,
        "lesionwise_dice_median": float(np.median(per_gt)),
        "lesion_detection_rate_dice_gt_0": float((per_gt > 0).mean()),
        "lesion_detection_rate_dice_ge_0_1": float((per_gt >= 0.1).mean()),
    })
    return out


def surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=STRUCT26, border_value=0)
    return mask & ~eroded


def hd95(gt: np.ndarray, pred: np.ndarray, spacing_xyz: Iterable[float]) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any() or not pred.any():
        return np.nan
    gt_surface = surface(gt)
    pred_surface = surface(pred)
    if not gt_surface.any() or not pred_surface.any():
        return np.nan
    sampling_zyx = tuple(reversed(tuple(float(x) for x in spacing_xyz)))
    dt_gt = ndimage.distance_transform_edt(~gt_surface, sampling=sampling_zyx)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=sampling_zyx)
    distances = np.concatenate([dt_pred[gt_surface], dt_gt[pred_surface]])
    if distances.size == 0:
        return np.nan
    return float(np.percentile(distances, 95))


def summarize_present(case_metrics: pd.DataFrame) -> pd.DataFrame:
    present = case_metrics[case_metrics["vol_bin"].isin(["small", "medium", "large"])]
    rows = []
    for (method, vol_bin), sub in present.groupby(["metodo", "vol_bin"], sort=False):
        rows.append({
            "metodo": method,
            "vol_bin": vol_bin,
            "n": len(sub),
            "global_dice_mean": sub["global_dice"].mean(),
            "global_dice_median": sub["global_dice"].median(),
            "lesionwise_dice_mean": sub["lesionwise_dice_mean"].mean(),
            "lesionwise_dice_median": sub["lesionwise_dice_mean"].median(),
            "lesion_detection_rate_mean": sub["lesion_detection_rate_dice_gt_0"].mean(),
            "hd95_median_defined": sub["hd95_mm"].median(skipna=True),
            "hd95_undefined_count": int(sub["hd95_mm"].isna().sum()),
            "overseg_ratio_median": sub["overseg_ratio"].median(),
            "n_global_dice_gt_0_4": int((sub["global_dice"] > 0.4).sum()),
            "n_lesionwise_dice_gt_0_4": int((sub["lesionwise_dice_mean"] > 0.4).sum()),
        })
    df = pd.DataFrame(rows)
    df["vol_bin"] = pd.Categorical(df["vol_bin"], VOL_ORDER, ordered=True)
    return df.sort_values(["metodo", "vol_bin"]).reset_index(drop=True)


def summarize_absent(case_metrics: pd.DataFrame) -> pd.DataFrame:
    absent = case_metrics[case_metrics["vol_bin"].eq("absent")]
    rows = []
    for method, sub in absent.groupby("metodo", sort=False):
        rows.append({
            "metodo": method,
            "n_absent": len(sub),
            "correct_absent_pred_lt_10_vox": int((sub["pred_vox"] < ABSENT_TOLERANCE_VOX).sum()),
            "correct_absent_rate": float((sub["pred_vox"] < ABSENT_TOLERANCE_VOX).mean()),
            "nonempty_pred_count": int((sub["pred_vox"] > 0).sum()),
            "fp_volume_median": sub["pred_vox"].median(),
            "fp_volume_p75": sub["pred_vox"].quantile(0.75),
            "fp_volume_p95": sub["pred_vox"].quantile(0.95),
            "fp_volume_max": sub["pred_vox"].max(),
            "large_fp_gt_1000_rate": float((sub["pred_vox"] > LARGE_FP_THRESHOLD_VOX).mean()),
            "flood_gt_10000_rate": float((sub["pred_vox"] > FLOOD_THRESHOLD_VOX).mean()),
        })
    return pd.DataFrame(rows).sort_values("metodo").reset_index(drop=True)


def summarize_fp_burden(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in case_metrics.groupby("metodo", sort=False):
        absent = sub[sub["vol_bin"].eq("absent")]
        present = sub[sub["vol_bin"].isin(["small", "medium", "large"])]
        rows.append({
            "metodo": method,
            "present_n": len(present),
            "present_lesionwise_dice_mean": present["lesionwise_dice_mean"].mean(),
            "present_lesionwise_dice_median": present["lesionwise_dice_mean"].median(),
            "present_global_dice_mean": present["global_dice"].mean(),
            "present_global_dice_median": present["global_dice"].median(),
            "present_overseg_ratio_median": present["overseg_ratio"].median(),
            "present_overseg_ratio_p75": present["overseg_ratio"].quantile(0.75),
            "present_overseg_ratio_max": present["overseg_ratio"].max(),
            "present_lesion_tp": int(present["lesion_tp"].sum()),
            "present_lesion_fn": int(present["lesion_fn"].sum()),
            "present_lesion_fp": int(present["lesion_fp"].sum()),
            "absent_n": len(absent),
            "absent_correct_rate_pred_lt_10": float((absent["pred_vox"] < ABSENT_TOLERANCE_VOX).mean()),
            "absent_fp_volume_median": absent["pred_vox"].median(),
            "absent_fp_volume_max": absent["pred_vox"].max(),
            "absent_flood_gt_10000_rate": float((absent["pred_vox"] > FLOOD_THRESHOLD_VOX).mean()),
        })
    df = pd.DataFrame(rows)
    df["rank_detection_lesionwise"] = df["present_lesionwise_dice_mean"].rank(
        ascending=False, method="min").astype(int)
    df["rank_fp_median_low_is_better"] = df["absent_fp_volume_median"].rank(
        ascending=True, method="min").astype(int)
    df["rank_flood_low_is_better"] = df["absent_flood_gt_10000_rate"].rank(
        ascending=True, method="min").astype(int)
    df["rank_sum"] = (
        df["rank_detection_lesionwise"]
        + df["rank_fp_median_low_is_better"]
        + df["rank_flood_low_is_better"]
    )
    return df.sort_values(["rank_sum", "rank_detection_lesionwise"]).reset_index(drop=True)


def overlap_check(case_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE)
    overlap = sorted(set(baseline["case_id"]) & set(case_metrics["case_id"]))
    if not overlap:
        return pd.DataFrame()
    base = baseline[
        baseline["case_id"].isin(overlap) & baseline["metodo"].isin(METHODS)
    ][["case_id", "metodo", "dice_ET", "vol_GT", "vol_pred"]]
    now = case_metrics[
        case_metrics["case_id"].isin(overlap) & case_metrics["metodo"].isin(METHODS)
    ][["case_id", "metodo", "global_dice", "gt_vox", "pred_vox"]]
    out = now.merge(base, on=["case_id", "metodo"], how="inner",
                    suffixes=("_stage4", "_baseline"))
    out["dice_delta"] = out["global_dice"] - out["dice_ET"]
    out["gt_delta"] = out["gt_vox"] - out["vol_GT"]
    out["pred_delta"] = out["pred_vox"] - out["vol_pred"]
    return out.sort_values(["case_id", "metodo"]).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    process = manifest[manifest["process"].astype(int).eq(1)].copy()
    process_cases = process["case_id"].astype(str).tolist()
    manifest_by_case = process.set_index("case_id").to_dict(orient="index")

    case_metrics_path = OUT / "stage4_case_metrics.csv"
    if case_metrics_path.exists() and "lesion_tp" in pd.read_csv(case_metrics_path, nrows=1).columns:
        print(f"[cache] loading {case_metrics_path}")
        case_metrics = pd.read_csv(case_metrics_path)
    else:
        rows: List[Dict[str, object]] = []
        for i, case_id in enumerate(process_cases, start=1):
            print(f"[{i:03d}/{len(process_cases)}] {case_id}", flush=True)
            seg_path = CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz"
            seg_arr, seg_img = load_mask(seg_path)
            gt = seg_arr == 3
            gt_vox = int(gt.sum())
            meta = manifest_by_case[case_id]

            for method in METHODS:
                pred_path = SEG_ROOT / case_id / f"{case_id}-et_{method}.nii.gz"
                pred_arr, _ = load_mask(pred_path)
                pred = pred_arr > 0
                pred_vox = int(pred.sum())
                row: Dict[str, object] = {
                    "case_id": case_id,
                    "metodo": method,
                    "vol_bin": meta["vol_bin"],
                    "focality": meta["focality"],
                    "fold": meta["fold"],
                    "et_present_manifest": int(meta["et_present"]),
                    "et_mm3_manifest": float(meta["et_mm3"]),
                    "gt_vox": gt_vox,
                    "pred_vox": pred_vox,
                    "global_dice": np.nan if gt_vox == 0 else dice(pred, gt),
                    "global_jaccard": np.nan if gt_vox == 0 else jaccard(pred, gt),
                    "overseg_ratio": np.nan if gt_vox == 0 else pred_vox / gt_vox,
                    "correct_absent_pred_lt_10_vox": (
                        bool(pred_vox < ABSENT_TOLERANCE_VOX) if gt_vox == 0 else np.nan
                    ),
                    "flood_gt_10000_vox": bool(pred_vox > FLOOD_THRESHOLD_VOX),
                }
                if gt_vox > 0:
                    row.update(lesionwise_dice(gt, pred))
                    row["hd95_mm"] = hd95(gt, pred, seg_img.GetSpacing())
                else:
                    row.update({
                        "gt_components": 0,
                        "pred_components": component_labels(pred)[1],
                        "matched_components": np.nan,
                        "lesion_tp": np.nan,
                        "lesion_fn": np.nan,
                        "lesion_fp": np.nan,
                        "lesion_dice_sum": np.nan,
                        "lesionwise_dice_mean": np.nan,
                        "lesionwise_dice_median": np.nan,
                        "lesion_detection_rate_dice_gt_0": np.nan,
                        "lesion_detection_rate_dice_ge_0_1": np.nan,
                        "hd95_mm": np.nan,
                    })
                rows.append(row)

        case_metrics = pd.DataFrame(rows)
        case_metrics.to_csv(case_metrics_path, index=False)

    present_by_vol = summarize_present(case_metrics)
    absent_fp = summarize_absent(case_metrics)
    fp_burden = summarize_fp_burden(case_metrics)
    overlap = overlap_check(case_metrics)

    present_by_vol.to_csv(OUT / "stage4_present_by_vol_bin.csv", index=False)
    absent_fp.to_csv(OUT / "stage4_absent_fp_summary.csv", index=False)
    fp_burden.to_csv(OUT / "stage4_method_ranking_detection_vs_fp.csv", index=False)
    overlap.to_csv(OUT / "stage4_overlap_global_dice_check.csv", index=False)

    headline = {
        "absent_tolerance_vox": ABSENT_TOLERANCE_VOX,
        "flood_threshold_vox": FLOOD_THRESHOLD_VOX,
        "large_fp_threshold_vox": LARGE_FP_THRESHOLD_VOX,
        "min_component_size_vox": MIN_COMPONENT_SIZE_VOX,
        "lesionwise_formula": (
            "sum(Dice over positive-overlap Hungarian matched components) / "
            "(TP positive matches + FN unmatched GT components + FP unmatched prediction components); "
            "GT and prediction components smaller than 10 voxels are filtered first."
        ),
        "n_process_cases": len(process_cases),
        "n_absent": int((process["vol_bin"] == "absent").sum()),
        "n_present": int((process["vol_bin"] != "absent").sum()),
        "outputs": {
            "case_metrics": str(OUT / "stage4_case_metrics.csv"),
            "present_by_vol_bin": str(OUT / "stage4_present_by_vol_bin.csv"),
            "absent_fp_summary": str(OUT / "stage4_absent_fp_summary.csv"),
            "method_ranking": str(OUT / "stage4_method_ranking_detection_vs_fp.csv"),
            "overlap_check": str(OUT / "stage4_overlap_global_dice_check.csv"),
        },
    }
    (OUT / "stage4_metric_config.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8")

    print("\nPRESENT BY VOLUME BIN")
    print(present_by_vol.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nABSENT FALSE POSITIVE SUMMARY")
    print(absent_fp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMETHOD RANKING")
    print(fp_burden.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOVERLAP CHECK")
    print(overlap.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
