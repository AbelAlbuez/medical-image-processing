"""Recompute and lock the Phase-2 metric targets on existing cohort masks.

Inputs: existing 100-case masks, cleaned GT segmentations, and the cohort
manifest. Outputs: baseline target CSVs and reproduction checks in ``phase2/``.
This script does not run segmentation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2.metrics import DEFAULT_CONFIG, score_et_case


RUN_ROOT = ROOT / "analysis" / "pre_cohort_run" / "cohort_100_core"
CLEAN_ROOT = RUN_ROOT / "limpieza"
SEG_ROOT = RUN_ROOT / "segmentacion"
MANIFEST = ROOT / "cohort" / "COHORT_MANIFEST_selected.csv"
STAGE4 = ROOT / "analysis" / "stage4_metrics"
OUT = ROOT / "phase2"
METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
VOL_ORDER = ["absent", "small", "medium", "large"]


def load_image(path: Path) -> tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def recompute_case_metrics() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    process = manifest[manifest["process"].astype(int).eq(1)].copy()
    meta_by_case = process.set_index("case_id").to_dict(orient="index")
    rows = []

    for idx, case_id in enumerate(process["case_id"].astype(str), start=1):
        print(f"[P0 metric] {idx:03d}/{len(process)} {case_id}", flush=True)
        seg_arr, seg_img = load_image(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
        gt_et = seg_arr == 3
        meta = meta_by_case[case_id]

        for method in METHODS:
            pred_arr, _ = load_image(SEG_ROOT / case_id / f"{case_id}-et_{method}.nii.gz")
            score = score_et_case(gt_et, pred_arr > 0, seg_img.GetSpacing())
            rows.append({
                "case_id": case_id,
                "metodo": method,
                "vol_bin": meta["vol_bin"],
                "focality": meta["focality"],
                "fold": meta["fold"],
                "et_present_manifest": int(meta["et_present"]),
                "et_mm3_manifest": float(meta["et_mm3"]),
                **score,
            })
    return pd.DataFrame(rows)


def compare_to_stage4(recomputed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    locked = pd.read_csv(STAGE4 / "stage4_case_metrics.csv")
    key = ["case_id", "metodo"]
    cols = [
        "gt_vox", "pred_vox", "global_dice", "global_jaccard", "overseg_ratio",
        "gt_components", "pred_components", "matched_components", "lesion_tp",
        "lesion_fn", "lesion_fp", "lesion_dice_sum", "lesionwise_dice_mean",
        "lesionwise_dice_median", "lesion_detection_rate_dice_gt_0",
        "lesion_detection_rate_dice_ge_0_1", "hd95_mm",
        "correct_absent_pred_lt_10_vox", "flood_gt_10000_vox",
    ]
    merged = recomputed[key + cols].merge(
        locked[key + cols], on=key, suffixes=("_recomputed", "_locked"),
        validate="one_to_one")
    diff_rows = []
    for col in cols:
        a = merged[f"{col}_recomputed"]
        b = merged[f"{col}_locked"]
        if a.dtype == object or b.dtype == object:
            different = a.astype(str).fillna("<NA>") != b.astype(str).fillna("<NA>")
            max_abs = np.nan
        else:
            av = pd.to_numeric(a, errors="coerce")
            bv = pd.to_numeric(b, errors="coerce")
            different = ~(
                np.isclose(av, bv, rtol=0.0, atol=1e-9, equal_nan=True)
            )
            max_abs = float(np.nanmax(np.abs(av - bv))) if different.any() else 0.0
        diff_rows.append({
            "column": col,
            "different_rows": int(different.sum()),
            "max_abs_diff": max_abs,
        })
    diff = pd.DataFrame(diff_rows)
    mismatch = diff[diff["different_rows"] > 0]
    return diff, mismatch


def summarize_present(case_metrics: pd.DataFrame) -> pd.DataFrame:
    present = case_metrics[case_metrics["vol_bin"].isin(["small", "medium", "large"])]
    rows = []
    for (method, vol_bin), sub in present.groupby(["metodo", "vol_bin"], sort=False):
        rows.append({
            "metodo": method,
            "vol_bin": vol_bin,
            "n": len(sub),
            "lesionwise_dice_mean": sub["lesionwise_dice_mean"].mean(),
            "lesionwise_dice_median": sub["lesionwise_dice_mean"].median(),
            "global_dice_mean": sub["global_dice"].mean(),
            "global_dice_median": sub["global_dice"].median(),
            "hd95_median_defined": sub["hd95_mm"].median(skipna=True),
            "lesion_tp": int(sub["lesion_tp"].sum()),
            "lesion_fn": int(sub["lesion_fn"].sum()),
            "lesion_fp": int(sub["lesion_fp"].sum()),
        })
    out = pd.DataFrame(rows)
    out["vol_bin"] = pd.Categorical(out["vol_bin"], VOL_ORDER, ordered=True)
    return out.sort_values(["metodo", "vol_bin"]).reset_index(drop=True)


def summarize_absent(case_metrics: pd.DataFrame) -> pd.DataFrame:
    absent = case_metrics[case_metrics["vol_bin"].eq("absent")]
    rows = []
    for method, sub in absent.groupby("metodo", sort=False):
        rows.append({
            "metodo": method,
            "n_absent": len(sub),
            "correct_absent_rate": float(sub["correct_absent_pred_lt_10_vox"].mean()),
            "fp_volume_median": sub["pred_vox"].median(),
            "fp_volume_max": sub["pred_vox"].max(),
            "flood_gt_10000_rate": float(sub["flood_gt_10000_vox"].mean()),
        })
    return pd.DataFrame(rows).sort_values("metodo").reset_index(drop=True)


def make_legacy_combined_ranking_note() -> pd.DataFrame:
    ranking = pd.read_csv(STAGE4 / "stage4_method_ranking_detection_vs_fp.csv")
    ranking = ranking.copy()
    ranking["legacy_combined_order"] = np.arange(1, len(ranking) + 1)
    ranking["legacy_formula"] = (
        "rank_sum = ordinal rank(present_lesionwise_dice_mean, descending) + "
        "ordinal rank(absent_fp_volume_median, ascending) + "
        "ordinal rank(absent_flood_gt_10000_rate, ascending); ties use min rank. "
        "This is an arbitrary equal-weight ordinal scalar and is NOT used as the Phase-2 baseline."
    )
    return ranking


def make_per_axis_baseline_targets(case_metrics: pd.DataFrame) -> pd.DataFrame:
    present = summarize_present(case_metrics)
    absent = summarize_absent(case_metrics)

    rows = []
    for _, row in present.iterrows():
        stratum = str(row["vol_bin"])
        peer = present[present["vol_bin"].astype(str).eq(stratum)].copy()
        best_value = peer["lesionwise_dice_mean"].max()
        rows.append({
            "axis": "detection",
            "target_scope": f"present_{stratum}",
            "metodo": row["metodo"],
            "is_axis_best": bool(np.isclose(row["lesionwise_dice_mean"], best_value)),
            "axis_primary_metric": "lesionwise_dice_mean",
            "axis_primary_direction": "higher_is_better",
            "n": int(row["n"]),
            "lesionwise_dice_mean": row["lesionwise_dice_mean"],
            "lesionwise_dice_median": row["lesionwise_dice_median"],
            "global_dice_mean": row["global_dice_mean"],
            "global_dice_median": row["global_dice_median"],
            "hd95_median_defined": row["hd95_median_defined"],
            "lesion_tp": row["lesion_tp"],
            "lesion_fn": row["lesion_fn"],
            "lesion_fp": row["lesion_fp"],
            "absent_flood_gt_10000_rate": np.nan,
            "absent_fp_volume_median": np.nan,
            "absent_fp_volume_max": np.nan,
        })

    absent_best_flood = absent["flood_gt_10000_rate"].min()
    candidates = absent[np.isclose(absent["flood_gt_10000_rate"], absent_best_flood)]
    absent_best_median = candidates["fp_volume_median"].min()
    for _, row in absent.iterrows():
        rows.append({
            "axis": "fp_restraint",
            "target_scope": "absent",
            "metodo": row["metodo"],
            "is_axis_best": bool(
                np.isclose(row["flood_gt_10000_rate"], absent_best_flood)
                and np.isclose(row["fp_volume_median"], absent_best_median)
            ),
            "axis_primary_metric": "absent_flood_gt_10000_rate_then_median_fp",
            "axis_primary_direction": "lower_is_better",
            "n": int(row["n_absent"]),
            "lesionwise_dice_mean": np.nan,
            "lesionwise_dice_median": np.nan,
            "global_dice_mean": np.nan,
            "global_dice_median": np.nan,
            "hd95_median_defined": np.nan,
            "lesion_tp": np.nan,
            "lesion_fn": np.nan,
            "lesion_fp": np.nan,
            "absent_flood_gt_10000_rate": row["flood_gt_10000_rate"],
            "absent_fp_volume_median": row["fp_volume_median"],
            "absent_fp_volume_max": row["fp_volume_max"],
        })
    return pd.DataFrame(rows)


def fold_counts() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    process = manifest[manifest["process"].astype(int).eq(1)].copy()
    rows = []
    for fold, sub in process.groupby("fold"):
        heldout_et_present = int((sub["vol_bin"] != "absent").sum())
        train = process[process["fold"] != fold]
        rows.append({
            "heldout_fold": int(fold),
            "heldout_cases": len(sub),
            "heldout_et_present": heldout_et_present,
            "heldout_absent": int((sub["vol_bin"] == "absent").sum()),
            "train_cases": len(train),
            "train_et_present": int((train["vol_bin"] != "absent").sum()),
            "train_absent": int((train["vol_bin"] == "absent").sum()),
            "train_small": int((train["vol_bin"] == "small").sum()),
            "train_medium": int((train["vol_bin"] == "medium").sum()),
            "train_large": int((train["vol_bin"] == "large").sum()),
        })
    return pd.DataFrame(rows).sort_values("heldout_fold").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recomputed_path = OUT / "p0_metric_recomputed_case_metrics.csv"
    if recomputed_path.exists():
        print(f"[cache] loading {recomputed_path}")
        recomputed = pd.read_csv(recomputed_path)
    else:
        recomputed = recompute_case_metrics()
        recomputed.to_csv(recomputed_path, index=False)

    diff, mismatch = compare_to_stage4(recomputed)
    diff.to_csv(OUT / "p0_metric_reproduction_diff.csv", index=False)

    legacy_ranking = make_legacy_combined_ranking_note()
    legacy_ranking.to_csv(OUT / "p0_legacy_combined_ranking_formula.csv", index=False)

    targets = make_per_axis_baseline_targets(recomputed)
    targets.to_csv(OUT / "baseline_targets_per_axis.csv", index=False)

    folds = fold_counts()
    folds.to_csv(OUT / "p0_fold_et_present_counts.csv", index=False)

    summary = pd.DataFrame([{
        "metric_reproduction_passed": bool(mismatch.empty),
        "mismatch_columns": "" if mismatch.empty else ",".join(mismatch["column"].astype(str)),
        "baseline_selection": "per-axis, not legacy combined scalar",
        "min_heldout_et_present": int(folds["heldout_et_present"].min()),
        "min_train_et_present": int(folds["train_et_present"].min()),
        "min_train_large": int(folds["train_large"].min()),
        "min_train_medium": int(folds["train_medium"].min()),
        "metric_formula": (
            "FP-aware lesion-wise Dice = sum positive-overlap Hungarian matched component Dice "
            "/ (TP + FN + FP), after filtering GT and prediction components <10 voxels. "
            "Absent cases use FP burden: pred volume, pred<10 vox absent-call, pred>10000 flood."
        ),
        "min_component_size_vox": DEFAULT_CONFIG.min_component_size_vox,
        "absent_tolerance_vox": DEFAULT_CONFIG.absent_tolerance_vox,
        "flood_threshold_vox": DEFAULT_CONFIG.flood_threshold_vox,
        "comparison_framing": (
            "Spatial priors target FP restraint: compare against the best absent-case FP baseline "
            "without regressing detection. Topological priors target detection: compare against the "
            "best per-stratum detection baseline without worsening FP."
        ),
    }])
    summary.to_csv(OUT / "p0_metric_lock_summary.csv", index=False)

    print("\nP0 METRIC REPRODUCTION")
    print(summary.to_string(index=False))
    print("\nDIFF")
    print(diff.to_string(index=False))
    print("\nLEGACY COMBINED RANKING FORMULA")
    print(legacy_ranking[[
        "metodo", "rank_detection_lesionwise", "rank_fp_median_low_is_better",
        "rank_flood_low_is_better", "rank_sum", "legacy_combined_order"
    ]].to_string(index=False))
    print("\nBASELINE TARGETS PER AXIS")
    print(targets.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nFOLD COUNTS")
    print(folds.to_string(index=False))
    if not mismatch.empty:
        raise SystemExit("Metric reproduction failed; see phase2/p0_metric_reproduction_diff.csv")


if __name__ == "__main__":
    main()
