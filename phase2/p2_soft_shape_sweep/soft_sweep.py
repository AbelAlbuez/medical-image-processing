"""Evaluate soft shape-proxy operating points on the locked Stage-4 metric.

Reads existing masks and applies train-fold learned component plausibility
scores. The sweep is diagnostic; reported operating points must be interpreted
with the leakage caveats in the audit report.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2.metrics import DEFAULT_CONFIG, component_labels


RUN_ROOT = ROOT / "analysis" / "pre_cohort_run" / "cohort_100_core"
CLEAN_ROOT = RUN_ROOT / "limpieza"
SEG_ROOT = RUN_ROOT / "segmentacion"
MANIFEST = ROOT / "cohort" / "COHORT_MANIFEST_selected.csv"
STAGE4_CASE_METRICS = ROOT / "analysis" / "stage4_metrics" / "stage4_case_metrics.csv"
BASELINE_TARGETS = ROOT / "phase2" / "baseline_targets_per_axis.csv"
P2A = ROOT / "phase2" / "p2_shape_prior"
COMPONENT_DECISIONS = P2A / "component_decisions"
P1_TRADEOFF = ROOT / "phase2" / "p1_spatial_atlas" / "tradeoff_analysis" / "p1_tradeoff_representative_curves.csv"
P1_OPS = ROOT / "phase2" / "p1_spatial_atlas" / "tradeoff_analysis" / "p1_fixed_detection_cost_operating_points.csv"
OUT = ROOT / "phase2" / "p2_soft_shape_sweep"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
VOL_ORDER = ["small", "medium", "large"]
SCORE_THRESHOLDS = [
    0.0, 0.001, 0.005, 0.01, 0.02, 0.05,
    0.10, 0.20, 0.30, 0.40, 0.50, 0.55,
]
LARGE_BASELINE = 0.1762380286972449
LARGE_90 = 0.90 * LARGE_BASELINE
OTSU_FLOOD = 0.9393939393939394
OTSU_MEDIAN_FP = 13_935.0
BOOTSTRAPS = 2000
RNG_SEED = 0


def read_arr(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def as_et(seg_arr: np.ndarray) -> np.ndarray:
    return np.round(seg_arr).astype(np.int16) == 3


def pred_path(case_id: str, method: str) -> Path:
    return SEG_ROOT / case_id / f"{case_id}-et_{method}.nii.gz"


def label_scores(mask: np.ndarray, decisions: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, n_labels = ndimage.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
    scores = np.full(n_labels + 1, -np.inf, dtype=np.float32)
    for _, row in decisions.iterrows():
        label = int(row["component_label"])
        if 0 <= label <= n_labels:
            scores[label] = float(row["shape_score_prob_true_et"])
    valid_metric = sizes >= DEFAULT_CONFIG.min_component_size_vox
    valid_metric[0] = False
    scores[~valid_metric] = -np.inf
    return labels, sizes.astype(np.int64), scores


def pair_intersections(gt_lab: np.ndarray, pred_lab: np.ndarray) -> pd.DataFrame:
    both = (gt_lab > 0) & (pred_lab > 0)
    if not np.any(both):
        return pd.DataFrame(columns=["gt_label", "pred_label", "intersection"])
    pairs = np.stack([gt_lab[both], pred_lab[both]], axis=1)
    pair_ids, counts = np.unique(pairs, axis=0, return_counts=True)
    return pd.DataFrame({
        "gt_label": pair_ids[:, 0].astype(int),
        "pred_label": pair_ids[:, 1].astype(int),
        "intersection": counts.astype(int),
    })


def score_from_components(
    gt_vox: int,
    gt_lab: np.ndarray,
    n_gt: int,
    gt_sizes: np.ndarray,
    pred_sizes: np.ndarray,
    pred_scores: np.ndarray,
    pairs: pd.DataFrame,
    threshold: float,
    vol_bin: str,
) -> Dict[str, object]:
    selected = np.flatnonzero(pred_scores >= threshold)
    selected = selected[selected != 0]
    pred_vox = int(pred_sizes[selected].sum()) if selected.size else 0
    out: Dict[str, object] = {
        "vol_bin": vol_bin,
        "gt_vox": gt_vox,
        "pred_vox": pred_vox,
        "flood_gt_10000_vox": bool(pred_vox > DEFAULT_CONFIG.flood_threshold_vox),
        "correct_absent_pred_lt_10_vox": (
            bool(pred_vox < DEFAULT_CONFIG.absent_tolerance_vox) if gt_vox == 0 else np.nan
        ),
    }
    if gt_vox == 0:
        out.update({
            "lesionwise_dice_mean": np.nan,
            "lesion_tp": np.nan,
            "lesion_fn": np.nan,
            "lesion_fp": np.nan,
            "dominant_found": np.nan,
        })
        return out

    n_pred = int(selected.size)
    if n_gt == 0:
        out.update({
            "lesionwise_dice_mean": np.nan,
            "lesion_tp": 0,
            "lesion_fn": 0,
            "lesion_fp": n_pred,
            "dominant_found": False,
        })
        return out
    if n_pred == 0:
        out.update({
            "lesionwise_dice_mean": 0.0,
            "lesion_tp": 0,
            "lesion_fn": int(n_gt),
            "lesion_fp": 0,
            "dominant_found": False,
        })
        return out

    pred_to_col = {int(label): idx for idx, label in enumerate(selected.tolist())}
    scores = np.zeros((n_gt, n_pred), dtype=np.float32)
    if len(pairs):
        sub = pairs[pairs["pred_label"].isin(pred_to_col.keys())]
        for _, row in sub.iterrows():
            gt_idx = int(row["gt_label"])
            pred_idx = int(row["pred_label"])
            if gt_idx <= 0 or gt_idx > n_gt:
                continue
            denom = gt_sizes[gt_idx] + pred_sizes[pred_idx]
            if denom > 0:
                scores[gt_idx - 1, pred_to_col[pred_idx]] = 2.0 * int(row["intersection"]) / denom
    gt_match_idx, pred_match_idx = linear_sum_assignment(-scores)
    matched_scores = scores[gt_match_idx, pred_match_idx]
    positive = matched_scores[matched_scores > 0]
    tp = int(positive.size)
    fn = int(n_gt - tp)
    fp = int(n_pred - tp)
    denom = tp + fn + fp
    lesionwise = float(positive.sum() / denom) if denom else np.nan
    out.update({
        "lesionwise_dice_mean": lesionwise,
        "lesion_tp": tp,
        "lesion_fn": fn,
        "lesion_fp": fp,
        "dominant_found": bool(tp > 0),
    })
    return out


def cache_matches_thresholds(case_csv: Path) -> bool:
    if not case_csv.exists():
        return False
    try:
        cached = pd.read_csv(case_csv, usecols=["shape_score_threshold", "score_engine"])
    except Exception:
        return False
    if not cached["score_engine"].eq("component_overlap_v1").all():
        return False
    got = sorted(float(x) for x in cached["shape_score_threshold"].dropna().unique())
    expected = sorted(float(x) for x in SCORE_THRESHOLDS)
    return got == expected


def evaluate_sweep(process: pd.DataFrame) -> pd.DataFrame:
    case_dir = OUT / "case_metrics"
    case_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx, meta in process.reset_index(drop=True).iterrows():
        case_id = str(meta["case_id"])
        case_csv = case_dir / f"{case_id}.csv"
        if cache_matches_thresholds(case_csv):
            print(f"[P2 soft] {idx+1:03d}/{len(process)} {case_id} [cache]", flush=True)
            frames.append(pd.read_csv(case_csv))
            continue
        print(f"[P2 soft] {idx+1:03d}/{len(process)} {case_id}", flush=True)
        vol_bin = str(meta["vol_bin"])
        seg = read_arr(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
        gt = as_et(seg)
        gt_vox = int(gt.sum())
        gt_lab, n_gt, gt_sizes = component_labels(gt, DEFAULT_CONFIG.min_component_size_vox)
        decisions_all = pd.read_csv(COMPONENT_DECISIONS / f"{case_id}.csv")
        rows: List[Dict[str, object]] = []
        for method in METHODS:
            pred = read_arr(pred_path(case_id, method)) > 0
            decisions = decisions_all[decisions_all["method"].eq(method)].copy()
            pred_lab, pred_sizes, pred_scores = label_scores(pred, decisions)
            pairs = pair_intersections(gt_lab, pred_lab) if gt_vox > 0 else pd.DataFrame()
            for threshold in SCORE_THRESHOLDS:
                rows.append({
                    "case_id": case_id,
                    "fold": int(meta["fold"]),
                    "method": method,
                    "mode": "soft_shape_score_threshold",
                    "shape_score_threshold": threshold,
                    "score_engine": "component_overlap_v1",
                    **score_from_components(
                        gt_vox, gt_lab, n_gt, gt_sizes,
                        pred_sizes, pred_scores, pairs, threshold, vol_bin),
                })
        case_df = pd.DataFrame(rows)
        case_df.to_csv(case_csv, index=False)
        frames.append(case_df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "p2_soft_shape_case_metrics.csv", index=False)
    return out


def aggregate(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, threshold), sub in case_metrics.groupby(["method", "shape_score_threshold"]):
        absent = sub[sub["vol_bin"].eq("absent")]
        row = {
            "method": method,
            "shape_score_threshold": threshold,
            "n_cases": sub["case_id"].nunique(),
            "absent_n": len(absent),
            "absent_flood_rate": float(absent["flood_gt_10000_vox"].mean()),
            "absent_median_fp_vox": float(absent["pred_vox"].median()),
            "absent_max_fp_vox": int(absent["pred_vox"].max()),
            "present_dominant_found_rate": float(sub[sub["vol_bin"].isin(VOL_ORDER)]["dominant_found"].mean()),
        }
        for vol in VOL_ORDER:
            s = sub[sub["vol_bin"].eq(vol)]
            row[f"{vol}_n"] = len(s)
            row[f"{vol}_lesionwise_mean"] = float(s["lesionwise_dice_mean"].mean())
            row[f"{vol}_dominant_found_rate"] = float(s["dominant_found"].mean())
            row[f"{vol}_lesion_tp"] = int(s["lesion_tp"].sum(skipna=True))
            row[f"{vol}_lesion_fn"] = int(s["lesion_fn"].sum(skipna=True))
            row[f"{vol}_lesion_fp"] = int(s["lesion_fp"].sum(skipna=True))
        row["large_detection_ratio"] = row["large_lesionwise_mean"] / LARGE_BASELINE
        row["keeps_large_detection_90pct"] = row["large_lesionwise_mean"] >= LARGE_90
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p2_soft_shape_sweep_summary.csv", index=False)
    return out


def detection_baseline_by_case() -> pd.DataFrame:
    stage4 = pd.read_csv(STAGE4_CASE_METRICS)
    targets = pd.read_csv(BASELINE_TARGETS)
    rows = []
    for _, row in targets[(targets["axis"].eq("detection")) & (targets["is_axis_best"])].iterrows():
        vol = str(row["target_scope"]).replace("present_", "")
        method = str(row["metodo"])
        sub = stage4[(stage4["vol_bin"].eq(vol)) & (stage4["metodo"].eq(method))].copy()
        sub["detection_baseline_method"] = method
        sub["detection_baseline_lesionwise"] = sub["lesionwise_dice_mean"]
        sub["detection_baseline_dominant_found"] = sub["lesion_tp"].fillna(0) > 0
        rows.append(sub[[
            "case_id", "vol_bin", "detection_baseline_method",
            "detection_baseline_lesionwise", "detection_baseline_dominant_found",
        ]])
    return pd.concat(rows, ignore_index=True)


def bootstrap_ci(values: Iterable[float], reducer=np.mean, n: int = BOOTSTRAPS) -> Tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(RNG_SEED)
    stats = []
    for _ in range(n):
        sample = values[rng.integers(0, values.size, size=values.size)]
        stats.append(float(reducer(sample)))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar_like_pvalue(delta_binary: np.ndarray) -> float:
    improved = int((delta_binary < 0).sum())
    worsened = int((delta_binary > 0).sum())
    discordant = improved + worsened
    if discordant == 0:
        return 1.0
    return float(binomtest(min(improved, worsened), discordant, 0.5).pvalue)


def key_table(case_metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    stage4 = pd.read_csv(STAGE4_CASE_METRICS)
    otsu_abs = stage4[(stage4["metodo"].eq("otsu_T1c")) & (stage4["vol_bin"].eq("absent"))]
    otsu_abs = otsu_abs[["case_id", "pred_vox", "flood_gt_10000_vox"]].rename(
        columns={"pred_vox": "otsu_pred_vox", "flood_gt_10000_vox": "otsu_flood"})
    det_base = detection_baseline_by_case()

    candidates = []
    for method, sub in summary.groupby("method"):
        ok = sub[sub["keeps_large_detection_90pct"]].copy()
        if ok.empty:
            chosen = sub.iloc[(sub["large_lesionwise_mean"] - LARGE_90).abs().argsort().iloc[0]]
            selection = "closest_to_90pct_no_config_preserves_90pct"
        else:
            chosen = ok.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
            selection = "lowest_fp_while_large_detection_ge_90pct"
        candidates.append(chosen.to_dict() | {"selection": selection})

    # Also include global best among any method satisfying the 90% gate.
    viable = summary[summary["keeps_large_detection_90pct"]].copy()
    if not viable.empty:
        candidates.append(
            viable.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0].to_dict()
            | {"selection": "global_best_fp_while_large_detection_ge_90pct"}
        )

    rows = []
    seen = set()
    for cand in candidates:
        key = (cand["method"], float(cand["shape_score_threshold"]), cand["selection"])
        if key in seen:
            continue
        seen.add(key)
        sub = case_metrics[
            case_metrics["method"].eq(cand["method"])
            & np.isclose(case_metrics["shape_score_threshold"], cand["shape_score_threshold"])
        ].copy()
        absent = sub[sub["vol_bin"].eq("absent")].merge(otsu_abs, on="case_id", how="inner")
        flood_delta = absent["flood_gt_10000_vox"].astype(float).to_numpy() - absent["otsu_flood"].astype(float).to_numpy()
        fp_delta = absent["pred_vox"].astype(float).to_numpy() - absent["otsu_pred_vox"].astype(float).to_numpy()
        row = dict(cand)
        row["delta_absent_flood_vs_otsu"] = float(np.mean(flood_delta))
        row["delta_absent_flood_ci_low"], row["delta_absent_flood_ci_high"] = bootstrap_ci(flood_delta)
        row["paired_flood_mcnemar_p"] = mcnemar_like_pvalue(flood_delta)
        row["delta_absent_median_fp_vs_otsu"] = float(absent["pred_vox"].median() - absent["otsu_pred_vox"].median())
        row["paired_absent_fp_vox_delta_median_ci_low"], row["paired_absent_fp_vox_delta_median_ci_high"] = bootstrap_ci(fp_delta, np.median)
        try:
            row["paired_fp_vox_wilcoxon_p"] = float(wilcoxon(fp_delta).pvalue)
        except ValueError:
            row["paired_fp_vox_wilcoxon_p"] = np.nan

        present = sub[sub["vol_bin"].isin(VOL_ORDER)].merge(det_base, on=["case_id", "vol_bin"], how="inner")
        for vol in VOL_ORDER:
            pv = present[present["vol_bin"].eq(vol)].copy()
            delta = (pv["lesionwise_dice_mean"] - pv["detection_baseline_lesionwise"]).to_numpy(float)
            row[f"{vol}_detection_baseline_mean"] = float(pv["detection_baseline_lesionwise"].mean())
            row[f"{vol}_delta_vs_detection_best"] = float(np.nanmean(delta)) if len(delta) else np.nan
            row[f"{vol}_delta_ci_low"], row[f"{vol}_delta_ci_high"] = bootstrap_ci(delta)
            try:
                row[f"{vol}_paired_wilcoxon_p"] = float(wilcoxon(delta).pvalue)
            except ValueError:
                row[f"{vol}_paired_wilcoxon_p"] = np.nan
            row[f"{vol}_baseline_dominant_found_rate"] = float(pv["detection_baseline_dominant_found"].mean())
            row[f"{vol}_dominant_found_delta"] = (
                row[f"{vol}_dominant_found_rate"] - row[f"{vol}_baseline_dominant_found_rate"]
            )
        row["clears_clean_bar"] = bool(
            row["keeps_large_detection_90pct"]
            and row["absent_flood_rate"] < OTSU_FLOOD
            and row["absent_median_fp_vox"] < OTSU_MEDIAN_FP
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p2_soft_shape_operating_points.csv", index=False)
    return out


def plot_tradeoff(summary: pd.DataFrame, ops: pd.DataFrame) -> None:
    p1 = pd.read_csv(P1_TRADEOFF)
    p1_vs = p1[p1["mode"].eq("post_filter") & p1["method"].eq("variational_spline")].sort_values("threshold")
    p2a = pd.read_csv(P2A / "p2_shape_key_comparison_table.csv")

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.plot(
        p1_vs["absent_flood_rate"],
        p1_vs["large_lesionwise_mean"],
        marker="o",
        linewidth=1.6,
        label="P1 spatial post-filter varspline",
    )
    for method, sub in summary.groupby("method"):
        sub = sub.sort_values("shape_score_threshold")
        alpha = 1.0 if method == "variational_spline" else 0.55
        ax.plot(
            sub["absent_flood_rate"],
            sub["large_lesionwise_mean"],
            marker=".",
            linewidth=1.2,
            alpha=alpha,
            label=f"P2 soft {method}",
        )
    ax.scatter(
        p2a["absent_flood_rate"],
        p2a["large_lesionwise_mean"],
        marker="x",
        s=60,
        color="black",
        label="P2a hard points",
    )
    ax.axhline(LARGE_BASELINE, color="black", linestyle="--", linewidth=1, label="large baseline")
    ax.axhline(LARGE_90, color="gray", linestyle=":", linewidth=1, label="90% large baseline")
    ax.axvline(OTSU_FLOOD, color="red", linestyle="--", linewidth=1, label="Otsu flood baseline")
    ax.set_xlabel("Absent-case flood rate")
    ax.set_ylabel("Large-stratum FP-aware lesion-wise Dice")
    ax.set_title("P2 soft shape prior vs P1 spatial prior")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(OUT / "p2_soft_shape_tradeoff_overlay.png", dpi=180)
    plt.close(fig)


def verdict(ops: pd.DataFrame) -> pd.DataFrame:
    p1_ops = pd.read_csv(P1_OPS)
    p1 = p1_ops[p1_ops["mode"].eq("post_filter") & p1_ops["method"].eq("variational_spline")].iloc[0]
    clean = ops[ops["clears_clean_bar"]].copy()
    viable = ops[ops["keeps_large_detection_90pct"]].copy()
    if not clean.empty:
        best = clean.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
        category = "a_soft_shape_usable_prior"
    elif not viable.empty:
        best = viable.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
        dominates_p1 = (
            best["absent_flood_rate"] < p1["absent_flood_rate"]
            and best["absent_median_fp_vox"] < p1["absent_median_fp_vox"]
            and best["large_detection_ratio"] >= 0.90
        )
        category = "b_shape_dominates_location_but_not_clean_bar" if dominates_p1 else "c_neither_shape_nor_location_clears_bar"
    else:
        best = ops.iloc[(ops["large_lesionwise_mean"] - LARGE_90).abs().argsort().iloc[0]]
        category = "c_neither_shape_nor_location_clears_bar"
    out = pd.DataFrame([{
        "verdict_category": category,
        "selected_method": best["method"],
        "selected_threshold": best["shape_score_threshold"],
        "absent_flood_rate": best["absent_flood_rate"],
        "absent_median_fp_vox": best["absent_median_fp_vox"],
        "large_lesionwise_mean": best["large_lesionwise_mean"],
        "large_detection_ratio": best["large_detection_ratio"],
        "p1_flood_at_90pct": p1["absent_flood_rate"],
        "p1_median_fp_at_90pct": p1["absent_median_fp_vox"],
        "p1_large_detection_ratio": p1["large_detection_ratio"],
        "clears_clean_bar": best["clears_clean_bar"],
    }])
    out.to_csv(OUT / "p2_soft_shape_verdict.csv", index=False)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    process = pd.read_csv(MANIFEST)
    process = process[process["process"].astype(int).eq(1)].copy()
    cases = evaluate_sweep(process)
    summary = aggregate(cases)
    ops = key_table(cases, summary)
    plot_tradeoff(summary, ops)
    ver = verdict(ops)

    print("\nOPERATING POINTS")
    cols = [
        "selection", "method", "shape_score_threshold", "absent_flood_rate",
        "absent_median_fp_vox", "delta_absent_flood_vs_otsu",
        "delta_absent_flood_ci_low", "delta_absent_flood_ci_high",
        "delta_absent_median_fp_vs_otsu", "large_lesionwise_mean",
        "large_detection_ratio", "large_dominant_found_rate", "clears_clean_bar",
    ]
    print(ops[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nVERDICT")
    print(ver.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nPlot: {OUT / 'p2_soft_shape_tradeoff_overlay.png'}")


if __name__ == "__main__":
    main()
