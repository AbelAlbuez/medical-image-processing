"""Evaluate the hard shape-proxy component filter on held-out folds.

The thresholds are learned on training folds and applied to held-out folds. This
uses normalized shape proxies, not genuine persistent homology.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.stats import binomtest, mannwhitneyu, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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
OLD20_MASK_ROOT = ROOT / "analysis" / "stage5_runner_validation" / "run_20260706_220739" / "segmentacion"
OLD20_DATA_ROOT = ROOT / "images"
OUT = ROOT / "phase2" / "p2_shape_prior"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
VOL_ORDER = ["small", "medium", "large"]
FEATURES = [
    "log10_isoperimetric",
    "radius_over_bbox_diag",
    "radius_over_equiv_sphere",
    "log1p_eroded_subcomponents",
]
PERI_CAVITY_CASES = ["BraTS-GLI-00533-100", "BraTS-GLI-02078-100"]
PERI_CAVITY_METHODS = ["gmm_T1c", "variational_spline"]
MIN_COMPONENT_SIZE = DEFAULT_CONFIG.min_component_size_vox
FLOOD_THRESHOLD = DEFAULT_CONFIG.flood_threshold_vox
BOOTSTRAPS = 2000
RNG_SEED = 0


def read_image(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def as_et(seg_arr: np.ndarray) -> np.ndarray:
    return np.round(seg_arr).astype(np.int16) == 3


def pred_path(case_id: str, method: str, root: Path = SEG_ROOT) -> Path:
    return root / case_id / f"{case_id}-et_{method}.nii.gz"


def exposed_face_surface(component: np.ndarray) -> float:
    padded = np.pad(component.astype(np.uint8), 1, mode="constant")
    dz = np.abs(np.diff(padded, axis=0)).sum()
    dy = np.abs(np.diff(padded, axis=1)).sum()
    dx = np.abs(np.diff(padded, axis=2)).sum()
    return float(dz + dy + dx)


def inscribed_radius_adaptive(component: np.ndarray) -> Tuple[float, int]:
    voxels = int(component.size)
    if voxels <= 350_000:
        return float(ndimage.distance_transform_edt(component).max()), 1
    step = 2 if voxels <= 2_000_000 else 4
    coarse = component[::step, ::step, ::step]
    return float(ndimage.distance_transform_edt(coarse).max() * step), step


def component_feature_rows(
    mask: np.ndarray,
    case_id: str,
    method: str,
    source: str,
    max_components: int | None = None,
    gt: np.ndarray | None = None,
) -> List[Dict[str, object]]:
    labels, n_labels = ndimage.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    if n_labels == 0:
        return []
    counts = np.bincount(labels.ravel())
    valid = np.flatnonzero(counts >= MIN_COMPONENT_SIZE)
    valid = valid[valid != 0]
    if len(valid) == 0:
        return []
    if max_components is not None and len(valid) > max_components:
        order = np.argsort(counts[valid])[::-1]
        valid = valid[order[:max_components]]
    valid_set = set(int(x) for x in valid)

    rows: List[Dict[str, object]] = []
    objects = ndimage.find_objects(labels)
    for label_idx, slc in enumerate(objects, start=1):
        if label_idx not in valid_set or slc is None:
            continue
        component = labels[slc] == label_idx
        volume = int(component.sum())
        extents = [int(s.stop - s.start) for s in slc]
        bbox_diag = float(np.sqrt(np.sum(np.square(extents))))
        max_extent = float(max(extents))
        min_extent = float(max(1, min(extents)))
        surface = exposed_face_surface(component)
        isoperimetric = (
            36.0 * np.pi * (volume**2) / (surface**3)
            if surface > 0 else np.nan
        )
        compactness = volume / (surface**1.5) if surface > 0 else np.nan
        radius, radius_step = inscribed_radius_adaptive(component)
        equiv_radius = ((3.0 * volume) / (4.0 * np.pi)) ** (1.0 / 3.0)
        eroded = ndimage.binary_erosion(component, structure=np.ones((3, 3, 3)), iterations=1)
        _, eroded_components = ndimage.label(eroded, structure=np.ones((3, 3, 3), dtype=bool))
        overlap_gt = np.nan
        if gt is not None:
            overlap_gt = int((component & (gt[slc] > 0)).sum())
        rows.append({
            "case_id": case_id,
            "method": method,
            "source": source,
            "component_label": label_idx,
            "volume_vox": volume,
            "surface_area_proxy": surface,
            "compactness_v_over_s15": compactness,
            "isoperimetric_quotient": isoperimetric,
            "inscribed_radius_vox": radius,
            "inscribed_radius_grid_step": radius_step,
            "bbox_diag_vox": bbox_diag,
            "radius_over_bbox_diag": radius / bbox_diag if bbox_diag > 0 else np.nan,
            "radius_over_equiv_sphere": radius / equiv_radius if equiv_radius > 0 else np.nan,
            "elongation_max_over_min": max_extent / min_extent,
            "eroded_subcomponents": int(eroded_components),
            "overlap_gt_vox": overlap_gt,
            "z0": slc[0].start,
            "z1": slc[0].stop,
            "y0": slc[1].start,
            "y1": slc[1].stop,
            "x0": slc[2].start,
            "x1": slc[2].stop,
        })
    return rows


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-12
    out["log10_isoperimetric"] = np.log10(pd.to_numeric(out["isoperimetric_quotient"], errors="coerce").clip(lower=eps))
    out["log1p_eroded_subcomponents"] = np.log1p(pd.to_numeric(out["eroded_subcomponents"], errors="coerce").clip(lower=0))
    return out


def training_components(process: pd.DataFrame, holdout_fold: int) -> pd.DataFrame:
    train = process[process["fold"].astype(int) != holdout_fold].copy()
    rows: List[Dict[str, object]] = []
    for _, meta in train.iterrows():
        case_id = str(meta["case_id"])
        seg, _ = read_image(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
        gt = as_et(seg)
        if str(meta["vol_bin"]) != "absent":
            rows.extend(component_feature_rows(gt, case_id, "GT_ET", "true_et_train"))
        else:
            for method in METHODS:
                pred, _ = read_image(pred_path(case_id, method))
                pred = pred > 0
                if int(pred.sum()) > FLOOD_THRESHOLD:
                    rows.extend(component_feature_rows(
                        pred, case_id, method, "absent_fp_train", max_components=1))
    df = add_model_features(pd.DataFrame(rows))
    df["label_true_et"] = df["source"].eq("true_et_train").astype(int)
    return df


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    candidates = np.unique(np.r_[0.0, prob, 1.0])
    best = None
    for threshold in candidates:
        pred = prob >= threshold
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        bal = 0.5 * (tpr + tnr)
        feasible = tpr >= 0.95
        record = {
            "threshold": float(threshold),
            "train_tpr": float(tpr),
            "train_tnr": float(tnr),
            "train_balanced_accuracy": float(bal),
            "train_fp_accept_rate": float(fp / max(1, fp + tn)),
            "train_fn_rate": float(fn / max(1, tp + fn)),
            "threshold_policy": "max_balanced_accuracy_with_tpr_ge_0.95",
            "feasible_tpr95": bool(feasible),
        }
        key = (feasible, bal, tnr, -threshold)
        if best is None or key > best[0]:
            best = (key, record)
    assert best is not None
    return best[1]


def fit_fold_model(train_df: pd.DataFrame) -> Tuple[StandardScaler, LogisticRegression, Dict[str, float]]:
    clean = train_df.dropna(subset=FEATURES + ["label_true_et"]).copy()
    x = clean[FEATURES].to_numpy(float)
    y = clean["label_true_et"].to_numpy(int)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = LogisticRegression(class_weight="balanced", random_state=RNG_SEED, max_iter=1000)
    model.fit(xs, y)
    prob = model.predict_proba(xs)[:, 1]
    info = choose_threshold(y, prob)
    info["train_auc"] = float(roc_auc_score(y, prob))
    info["train_n_true_et_components"] = int(y.sum())
    info["train_n_fp_components"] = int((y == 0).sum())
    for feature, coef in zip(FEATURES, model.coef_[0]):
        info[f"coef_{feature}"] = float(coef)
    info["intercept"] = float(model.intercept_[0])
    return scaler, model, info


def filter_mask_by_shape(
    mask: np.ndarray,
    case_id: str,
    method: str,
    scaler: StandardScaler,
    model: LogisticRegression,
    threshold: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    labels, n_labels = ndimage.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool), pd.DataFrame()
    rows = component_feature_rows(mask, case_id, method, "prediction_component")
    if not rows:
        return np.zeros_like(mask, dtype=bool), pd.DataFrame()
    df = add_model_features(pd.DataFrame(rows))
    valid = df.dropna(subset=FEATURES).copy()
    if len(valid):
        prob = model.predict_proba(scaler.transform(valid[FEATURES].to_numpy(float)))[:, 1]
        valid["shape_score_prob_true_et"] = prob
        valid["accepted_shape"] = valid["shape_score_prob_true_et"] >= threshold
        df = df.merge(
            valid[["component_label", "shape_score_prob_true_et", "accepted_shape"]],
            on="component_label",
            how="left",
        )
    else:
        df["shape_score_prob_true_et"] = np.nan
        df["accepted_shape"] = False
    df["accepted_shape"] = df["accepted_shape"].fillna(False).astype(bool)
    keep_labels = set(df.loc[df["accepted_shape"], "component_label"].astype(int).tolist())
    keep = np.zeros(n_labels + 1, dtype=bool)
    for label in keep_labels:
        if 0 <= label <= n_labels:
            keep[label] = True
    return keep[labels], df


def score_fast(gt: np.ndarray, pred: np.ndarray, vol_bin: str) -> Dict[str, object]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_vox = int(gt.sum())
    pred_vox = int(pred.sum())
    out: Dict[str, object] = {
        "vol_bin": vol_bin,
        "gt_vox": gt_vox,
        "pred_vox": pred_vox,
        "flood_gt_10000_vox": bool(pred_vox > FLOOD_THRESHOLD),
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
    lesion = fp_aware_lesionwise_dice(gt, pred, DEFAULT_CONFIG.min_component_size_vox)
    out.update({
        "lesionwise_dice_mean": lesion["lesionwise_dice_mean"],
        "lesion_tp": lesion["lesion_tp"],
        "lesion_fn": lesion["lesion_fn"],
        "lesion_fp": lesion["lesion_fp"],
        "dominant_found": bool(lesion["lesion_tp"] > 0),
    })
    return out


def evaluate_shape_filter(process: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_dir = OUT / "case_metrics"
    comp_dir = OUT / "component_decisions"
    case_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)
    threshold_rows = []
    case_frames = []
    component_frames = []
    models: Dict[int, Tuple[StandardScaler, LogisticRegression, Dict[str, float]]] = {}
    for fold in sorted(process["fold"].dropna().astype(int).unique()):
        train_df = training_components(process, int(fold))
        scaler, model, info = fit_fold_model(train_df)
        info["holdout_fold"] = int(fold)
        threshold_rows.append(info)
        models[int(fold)] = (scaler, model, info)
        train_df.to_csv(OUT / f"p2_train_components_holdout_fold{fold}.csv", index=False)

    for idx, meta in process.reset_index(drop=True).iterrows():
        case_id = str(meta["case_id"])
        case_csv = case_dir / f"{case_id}.csv"
        comp_csv = comp_dir / f"{case_id}.csv"
        if case_csv.exists() and comp_csv.exists():
            print(f"[P2 eval] {idx+1:03d}/{len(process)} {case_id} [cache]", flush=True)
            case_frames.append(pd.read_csv(case_csv))
            component_frames.append(pd.read_csv(comp_csv))
            continue
        fold = int(meta["fold"])
        scaler, model, info = models[fold]
        threshold = float(info["threshold"])
        vol_bin = str(meta["vol_bin"])
        seg, _ = read_image(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
        gt = as_et(seg)
        print(f"[P2 eval] {idx+1:03d}/{len(process)} {case_id}", flush=True)
        rows = []
        comp_rows = []
        for method in METHODS:
            pred, _ = read_image(pred_path(case_id, method))
            pred = pred > 0
            filtered, comp = filter_mask_by_shape(pred, case_id, method, scaler, model, threshold)
            rows.append({
                "case_id": case_id,
                "fold": fold,
                "method": method,
                "mode": "shape_post_filter",
                "shape_threshold": threshold,
                **score_fast(gt, filtered, vol_bin),
            })
            if len(comp):
                comp["case_id"] = case_id
                comp["fold"] = fold
                comp["method"] = method
                comp["vol_bin"] = vol_bin
                comp["shape_threshold"] = threshold
                comp_rows.append(comp)
        case_df = pd.DataFrame(rows)
        comp_df = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame()
        case_df.to_csv(case_csv, index=False)
        comp_df.to_csv(comp_csv, index=False)
        case_frames.append(case_df)
        component_frames.append(comp_df)

    cases = pd.concat(case_frames, ignore_index=True)
    comps = pd.concat(component_frames, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    cases.to_csv(OUT / "p2_shape_filter_case_metrics.csv", index=False)
    comps.to_csv(OUT / "p2_shape_component_decisions.csv", index=False)
    thresholds.to_csv(OUT / "p2_shape_thresholds_by_fold.csv", index=False)
    return cases, comps, thresholds


def aggregate(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in case_metrics.groupby("method"):
        absent = sub[sub["vol_bin"].eq("absent")]
        row = {
            "method": method,
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
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p2_shape_filter_summary.csv", index=False)
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
    # delta = filtered_flood - baseline_flood; -1 improves, +1 worsens.
    improved = int((delta_binary < 0).sum())
    worsened = int((delta_binary > 0).sum())
    discordant = improved + worsened
    if discordant == 0:
        return 1.0
    return float(binomtest(min(improved, worsened), discordant, 0.5).pvalue)


def key_comparison(case_metrics: pd.DataFrame) -> pd.DataFrame:
    stage4 = pd.read_csv(STAGE4_CASE_METRICS)
    otsu_abs = stage4[(stage4["metodo"].eq("otsu_T1c")) & (stage4["vol_bin"].eq("absent"))]
    otsu_abs = otsu_abs[["case_id", "pred_vox", "flood_gt_10000_vox"]].rename(
        columns={"pred_vox": "otsu_pred_vox", "flood_gt_10000_vox": "otsu_flood"})
    det_base = detection_baseline_by_case()

    rows = []
    for method, sub in case_metrics.groupby("method"):
        row = {"method": method}
        absent = sub[sub["vol_bin"].eq("absent")].merge(otsu_abs, on="case_id", how="inner")
        flood_delta = absent["flood_gt_10000_vox"].astype(float).to_numpy() - absent["otsu_flood"].astype(float).to_numpy()
        fp_delta = absent["pred_vox"].astype(float).to_numpy() - absent["otsu_pred_vox"].astype(float).to_numpy()
        row["absent_flood_rate"] = float(absent["flood_gt_10000_vox"].mean())
        row["absent_median_fp_vox"] = float(absent["pred_vox"].median())
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
            row[f"{vol}_lesionwise_mean"] = float(pv["lesionwise_dice_mean"].mean())
            row[f"{vol}_detection_baseline_mean"] = float(pv["detection_baseline_lesionwise"].mean())
            row[f"{vol}_retention_vs_detection_baseline"] = (
                row[f"{vol}_lesionwise_mean"] / row[f"{vol}_detection_baseline_mean"]
                if row[f"{vol}_detection_baseline_mean"] else np.nan
            )
            row[f"{vol}_delta_vs_detection_best"] = float(np.nanmean(delta))
            row[f"{vol}_delta_ci_low"], row[f"{vol}_delta_ci_high"] = bootstrap_ci(delta)
            try:
                row[f"{vol}_paired_wilcoxon_p"] = float(wilcoxon(delta).pvalue)
            except ValueError:
                row[f"{vol}_paired_wilcoxon_p"] = np.nan
            row[f"{vol}_dominant_found_rate"] = float(pv["dominant_found"].mean())
            row[f"{vol}_baseline_dominant_found_rate"] = float(pv["detection_baseline_dominant_found"].mean())
            row[f"{vol}_dominant_found_delta"] = (
                row[f"{vol}_dominant_found_rate"] - row[f"{vol}_baseline_dominant_found_rate"]
            )
        row["large_detection_preserved_90pct"] = bool(row["large_retention_vs_detection_baseline"] >= 0.90)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p2_shape_key_comparison_table.csv", index=False)
    return out


def peri_cavity_check(thresholds: pd.DataFrame) -> pd.DataFrame:
    # Corroborating only: these two old-20 cases are not part of the 100-case CV split.
    rows = []
    threshold = float(thresholds["threshold"].median())
    train_df = pd.concat(
        [pd.read_csv(path) for path in sorted(OUT.glob("p2_train_components_holdout_fold*.csv"))],
        ignore_index=True,
    )
    scaler, model, _ = fit_fold_model(train_df)
    for case_id in PERI_CAVITY_CASES:
        seg, _ = read_image(OLD20_DATA_ROOT / case_id / f"{case_id}-seg.nii.gz")
        gt = as_et(seg)
        for method in PERI_CAVITY_METHODS:
            pred, _ = read_image(pred_path(case_id, method, OLD20_MASK_ROOT))
            pred = pred > 0
            _, comp = filter_mask_by_shape(pred, case_id, method, scaler, model, threshold)
            if len(comp):
                comp["fp_component"] = comp["overlap_gt_vox"].fillna(0).eq(0)
                rows.append({
                    "case_id": case_id,
                    "method": method,
                    "n_components": len(comp),
                    "n_fp_components": int(comp["fp_component"].sum()),
                    "n_fp_rejected": int((comp["fp_component"] & ~comp["accepted_shape"]).sum()),
                    "n_fp_accepted": int((comp["fp_component"] & comp["accepted_shape"]).sum()),
                    "threshold_used": threshold,
                    "min_fp_shape_score": float(comp.loc[comp["fp_component"], "shape_score_prob_true_et"].min()),
                    "max_fp_shape_score": float(comp.loc[comp["fp_component"], "shape_score_prob_true_et"].max()),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p2_peri_cavity_shape_check.csv", index=False)
    return out


def p1_vs_p2_table(key: pd.DataFrame) -> pd.DataFrame:
    p1 = pd.read_csv(ROOT / "phase2" / "p1_spatial_atlas" / "tradeoff_analysis" / "p1_fixed_detection_cost_operating_points.csv")
    p1_best = p1[(p1["mode"].eq("post_filter")) & (p1["method"].eq("variational_spline"))].iloc[0]
    viable = key[key["large_detection_preserved_90pct"]].copy()
    if viable.empty:
        p2_best = key.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
        selection = "best_fp_no_90pct_detection_candidate"
    else:
        p2_best = viable.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
        selection = "best_fp_with_large_detection_ge_90pct"
    out = pd.DataFrame([
        {
            "prior": "P1 spatial post-filter",
            "method": p1_best["method"],
            "selection": "P1 fixed 90pct detection operating point",
            "absent_flood_rate": p1_best["absent_flood_rate"],
            "absent_median_fp_vox": p1_best["absent_median_fp_vox"],
            "large_lesionwise": p1_best["large_lesionwise"],
            "large_detection_ratio": p1_best["large_detection_ratio"],
            "large_dominant_found_rate": p1_best["large_dominant_found_rate"],
        },
        {
            "prior": "P2 shape post-filter",
            "method": p2_best["method"],
            "selection": selection,
            "absent_flood_rate": p2_best["absent_flood_rate"],
            "absent_median_fp_vox": p2_best["absent_median_fp_vox"],
            "large_lesionwise": p2_best["large_lesionwise_mean"],
            "large_detection_ratio": p2_best["large_retention_vs_detection_baseline"],
            "large_dominant_found_rate": p2_best["large_dominant_found_rate"],
        },
    ])
    out.to_csv(OUT / "p2_vs_p1_fixed_detection_comparison.csv", index=False)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    process = pd.read_csv(MANIFEST)
    process = process[process["process"].astype(int).eq(1)].copy()

    cases, comps, thresholds = evaluate_shape_filter(process)
    summary = aggregate(cases)
    key = key_comparison(cases)
    peri = peri_cavity_check(thresholds)
    p1p2 = p1_vs_p2_table(key)

    print("\nTHRESHOLDS")
    print(thresholds[[
        "holdout_fold", "threshold", "train_auc", "train_tpr", "train_tnr",
        "train_fp_accept_rate", "train_n_true_et_components", "train_n_fp_components",
    ]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nKEY COMPARISON")
    cols = [
        "method", "absent_flood_rate", "absent_median_fp_vox",
        "delta_absent_flood_vs_otsu", "delta_absent_flood_ci_low", "delta_absent_flood_ci_high",
        "delta_absent_median_fp_vs_otsu", "large_lesionwise_mean",
        "large_retention_vs_detection_baseline", "large_dominant_found_rate",
        "large_detection_preserved_90pct",
    ]
    print(key[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nP1 VS P2")
    print(p1p2.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nPERI-CAVITY CHECK (n=4 corroborating only)")
    print(peri.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
