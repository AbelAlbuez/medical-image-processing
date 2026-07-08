"""Run genuine GUDHI cubical H0 persistence diagnostics and count/rank priors.

This is the only Phase 2 topological computation: it computes cubical persistent
homology on normalized enhancement-map crops. It reads existing masks/volumes
and writes P3 diagnostic and held-out comparison tables.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys
import time
from typing import Dict, Iterable, List, Tuple

import gudhi as gd
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2.metrics import DEFAULT_CONFIG, component_labels, dice, fp_aware_lesionwise_dice, jaccard


RUN_ROOT = ROOT / "analysis" / "pre_cohort_run" / "cohort_100_core"
CLEAN_ROOT = RUN_ROOT / "limpieza"
SEG_ROOT = RUN_ROOT / "segmentacion"
MANIFEST = ROOT / "cohort" / "COHORT_MANIFEST_selected.csv"
STAGE4_CASE_METRICS = ROOT / "analysis" / "stage4_metrics" / "stage4_case_metrics.csv"
BASELINE_TARGETS = ROOT / "phase2" / "baseline_targets_per_axis.csv"
P2B_OPS = ROOT / "phase2" / "p2_soft_shape_sweep" / "p2_soft_shape_operating_points.csv"
IRREDUCIBLE = ROOT / "analysis" / "stage3d_irreducible_mechanism.csv"
OLD20_MASK_ROOT = ROOT / "analysis" / "stage5_runner_validation" / "run_20260706_220739" / "segmentacion"
OLD20_DATA_ROOT = ROOT / "images"
OUT = ROOT / "phase2" / "p3_cubical_persistence"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
METHOD_TYPES = {
    "otsu_T1c": "intensity/statistical",
    "gmm_T1c": "intensity/statistical",
    "gmm_2d": "intensity/statistical",
    "sustraccion": "intensity/statistical",
    "variational_spline": "deformable/region-based",
}
VOL_ORDER = ["small", "medium", "large"]
STRUCT26 = np.ones((3, 3, 3), dtype=bool)

# This is a diagnostic PH pass, not a production segmenter. To keep runtime bounded,
# prediction masks with thousands of components are represented by the union of their
# largest components and their brightest components. Components outside this candidate
# set are rejected by the top-k prior.
CANDIDATE_BY_VOLUME = 5
CANDIDATE_BY_MAX_ENH = 5
MAX_K = 5
PH_PAD_VOX = 2
MAX_CROP_VOXELS_EXACT = 300_000
BOOTSTRAPS = 2000
RNG_SEED = 0


def read_image(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def as_et(seg_arr: np.ndarray) -> np.ndarray:
    return np.round(seg_arr).astype(np.int16) == 3


def pred_path(case_id: str, method: str, root: Path = SEG_ROOT) -> Path:
    return root / case_id / f"{case_id}-et_{method}.nii.gz"


def enhancement_score(case_id: str, root: Path = CLEAN_ROOT) -> np.ndarray:
    t1c, _ = read_image(root / case_id / f"{case_id}-t1c.nii.gz")
    t1n, _ = read_image(root / case_id / f"{case_id}-t1n.nii.gz")
    diff = np.maximum(t1c.astype(np.float32) - t1n.astype(np.float32), 0.0)
    positive = diff[diff > 0]
    if positive.size == 0:
        return np.zeros_like(diff, dtype=np.float32)
    scale = float(np.percentile(positive, 99.5))
    if scale <= 0:
        return np.zeros_like(diff, dtype=np.float32)
    return np.clip(diff / scale, 0.0, 1.0).astype(np.float32)


def crop_with_pad(slc: Tuple[slice, slice, slice],
                  shape: Tuple[int, int, int],
                  pad: int = PH_PAD_VOX) -> Tuple[slice, slice, slice]:
    return tuple(
        slice(max(0, s.start - pad), min(limit, s.stop + pad))
        for s, limit in zip(slc, shape)
    )


def downsample_for_ph(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
    if values.size <= MAX_CROP_VOXELS_EXACT:
        return values, 1
    step = int(math.ceil((values.size / MAX_CROP_VOXELS_EXACT) ** (1.0 / 3.0)))
    step = max(2, step)
    ds_values = values[::step, ::step, ::step]
    ds_mask = mask[::step, ::step, ::step]
    return np.where(ds_mask, ds_values, 0.0), step


def h0_persistence_from_component(score: np.ndarray,
                                  labels: np.ndarray,
                                  label: int,
                                  objects: List[Tuple[slice, slice, slice] | None]) -> Dict[str, object]:
    slc = objects[label - 1]
    if slc is None:
        return {
            "ph_error": "missing_slice",
            "ph_h0_max_persistence": np.nan,
            "ph_h0_second_persistence": np.nan,
            "ph_h0_n_ge_0_10": np.nan,
            "ph_h1_max_persistence": np.nan,
            "ph_h2_max_persistence": np.nan,
            "ph_downsample_step": np.nan,
            "ph_crop_voxels": 0,
            "ph_runtime_s": 0.0,
        }
    slc_pad = crop_with_pad(slc, score.shape)
    component = labels[slc_pad] == label
    values = np.where(component, score[slc_pad], 0.0).astype(np.float32)
    values, step = downsample_for_ph(values, component)
    start = time.perf_counter()
    try:
        # Superlevel-set PH of enhancement is computed as sublevel-set PH on -enhancement.
        # Values are already normalized to [0, 1], so H0 persistence is a normalized
        # peak-prominence score in [0, 1] after capping the essential class at background.
        filtration = -values.astype(np.float64)
        cc = gd.CubicalComplex(
            dimensions=filtration.shape,
            top_dimensional_cells=filtration.ravel(order="F"),
        )
        diagram = cc.persistence(homology_coeff_field=2, min_persistence=0.0)
        max_filtration = float(filtration.max())
        by_dim: Dict[int, List[float]] = {0: [], 1: [], 2: []}
        for dim, (birth, death) in diagram:
            if dim not in by_dim:
                continue
            if np.isinf(death):
                death = max_filtration
            persistence = max(0.0, float(death - birth))
            if persistence > 0:
                by_dim[dim].append(persistence)
        for dim in by_dim:
            by_dim[dim].sort(reverse=True)
        runtime = time.perf_counter() - start
        return {
            "ph_error": "",
            "ph_h0_max_persistence": by_dim[0][0] if by_dim[0] else 0.0,
            "ph_h0_second_persistence": by_dim[0][1] if len(by_dim[0]) > 1 else 0.0,
            "ph_h0_n_ge_0_10": int(sum(x >= 0.10 for x in by_dim[0])),
            "ph_h1_max_persistence": by_dim[1][0] if by_dim[1] else 0.0,
            "ph_h2_max_persistence": by_dim[2][0] if by_dim[2] else 0.0,
            "ph_downsample_step": int(step),
            "ph_crop_voxels": int(values.size),
            "ph_runtime_s": runtime,
        }
    except Exception as exc:  # pragma: no cover - diagnostic output path
        return {
            "ph_error": repr(exc),
            "ph_h0_max_persistence": np.nan,
            "ph_h0_second_persistence": np.nan,
            "ph_h0_n_ge_0_10": np.nan,
            "ph_h1_max_persistence": np.nan,
            "ph_h2_max_persistence": np.nan,
            "ph_downsample_step": int(step),
            "ph_crop_voxels": int(values.size),
            "ph_runtime_s": time.perf_counter() - start,
        }


def component_candidates(labels: np.ndarray, score: np.ndarray, sizes: np.ndarray) -> List[int]:
    valid = np.flatnonzero(sizes >= DEFAULT_CONFIG.min_component_size_vox)
    valid = valid[valid != 0]
    if valid.size == 0:
        return []
    by_volume = valid[np.argsort(sizes[valid])[::-1][:CANDIDATE_BY_VOLUME]]
    maxima = ndimage.maximum(score, labels, index=valid)
    by_enh = valid[np.argsort(maxima)[::-1][:CANDIDATE_BY_MAX_ENH]]
    ordered: List[int] = []
    for label in list(by_volume) + list(by_enh):
        label = int(label)
        if label not in ordered:
            ordered.append(label)
    return ordered


def component_rows(mask: np.ndarray,
                   score: np.ndarray,
                   case_id: str,
                   method: str,
                   source: str,
                   candidate_only: bool = True) -> List[Dict[str, object]]:
    labels, n_labels, sizes = component_labels(mask, DEFAULT_CONFIG.min_component_size_vox)
    if n_labels == 0:
        return []
    labels_to_score = component_candidates(labels, score, sizes) if candidate_only else list(range(1, n_labels + 1))
    objects = ndimage.find_objects(labels)
    rows: List[Dict[str, object]] = []
    for label in labels_to_score:
        label = int(label)
        slc = objects[label - 1]
        if slc is None:
            continue
        component = labels[slc] == label
        volume = int(component.sum())
        enh_vals = score[slc][component]
        record = {
            "case_id": case_id,
            "method": method,
            "source": source,
            "component_label": label,
            "component_rank_by_volume": int(1 + np.sum(sizes[1:] > sizes[label])),
            "volume_vox": volume,
            "enh_max": float(enh_vals.max()) if enh_vals.size else 0.0,
            "enh_mean": float(enh_vals.mean()) if enh_vals.size else 0.0,
            "candidate_policy": (
                f"top{CANDIDATE_BY_VOLUME}_volume_union_top{CANDIDATE_BY_MAX_ENH}_enh"
                if candidate_only else "all_components"
            ),
            "n_metric_components_in_mask": int(n_labels),
        }
        record.update(h0_persistence_from_component(score, labels, label, objects))
        rows.append(record)
    return rows


def build_component_persistence(process: pd.DataFrame) -> pd.DataFrame:
    out_path = OUT / "p3_component_persistence.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    cache_dir = OUT / "component_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    for idx, meta in process.reset_index(drop=True).iterrows():
        case_id = str(meta["case_id"])
        cache_path = cache_dir / f"{case_id}.csv"
        if cache_path.exists():
            print(f"[P3 PH] cohort {idx + 1:03d}/{len(process)} {case_id} [cache]", flush=True)
            frames.append(pd.read_csv(cache_path))
            continue
        print(f"[P3 PH] cohort {idx + 1:03d}/{len(process)} {case_id}", flush=True)
        rows: List[Dict[str, object]] = []
        score = enhancement_score(case_id)
        seg, _ = read_image(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
        gt = as_et(seg)
        if str(meta["vol_bin"]) != "absent":
            rows.extend(component_rows(
                gt, score, case_id, "GT_ET", "true_et_cohort", candidate_only=False))
        for method in METHODS:
            pred, _ = read_image(pred_path(case_id, method))
            pred = pred > 0
            source = "prediction_component"
            if str(meta["vol_bin"]) == "absent" and int(pred.sum()) > DEFAULT_CONFIG.flood_threshold_vox:
                source = "absent_fp_component"
            rows.extend(component_rows(pred, score, case_id, method, source, candidate_only=True))
        case_df = pd.DataFrame(rows)
        case_df.to_csv(cache_path, index=False)
        frames.append(case_df)

    # Corroborating old-20 peri-cavity confounds. These are outside the 100-case CV split.
    peri_cache = cache_dir / "_peri_cavity_old20.csv"
    if peri_cache.exists():
        frames.append(pd.read_csv(peri_cache))
    else:
        peri_rows: List[Dict[str, object]] = []
        for case_id in ["BraTS-GLI-00533-100", "BraTS-GLI-02078-100"]:
            print(f"[P3 PH] peri-cavity old20 {case_id}", flush=True)
            score = enhancement_score(case_id, OLD20_DATA_ROOT)
            seg, _ = read_image(OLD20_DATA_ROOT / case_id / f"{case_id}-seg.nii.gz")
            gt = as_et(seg)
            peri_rows.extend(component_rows(
                gt, score, case_id, "GT_ET", "true_et_old20", candidate_only=False))
            for method in ["gmm_T1c", "variational_spline"]:
                pred, _ = read_image(pred_path(case_id, method, OLD20_MASK_ROOT))
                pred = pred > 0
                pred_labels, _, pred_sizes = component_labels(pred, DEFAULT_CONFIG.min_component_size_vox)
                gt_overlap = ndimage.sum(gt.astype(np.uint8), pred_labels, index=np.arange(1, len(pred_sizes)))
                fp_mask = np.isin(pred_labels, np.flatnonzero(gt_overlap == 0) + 1)
                peri_rows.extend(component_rows(
                    fp_mask, score, case_id, method, "peri_cavity_fp_component", candidate_only=True))
        peri_df = pd.DataFrame(peri_rows)
        peri_df.to_csv(peri_cache, index=False)
        frames.append(peri_df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(out_path, index=False)
    return df


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


def persistence_diagnostic(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    true = components[components["source"].isin(["true_et_cohort", "true_et_old20"])].copy()
    groups = {
        "absent_fp": components[components["source"].eq("absent_fp_component")],
        "peri_cavity_fp_n4": components[components["source"].eq("peri_cavity_fp_component")],
        "all_confound": components[components["source"].isin(["absent_fp_component", "peri_cavity_fp_component"])],
    }
    for name, conf in groups.items():
        score_col = "ph_h0_max_persistence"
        sub = pd.concat([
            true.assign(label_true_et=1),
            conf.assign(label_true_et=0),
        ], ignore_index=True)
        sub = sub[np.isfinite(sub[score_col])]
        if sub["label_true_et"].nunique() < 2:
            auc = np.nan
            p = np.nan
        else:
            auc = float(roc_auc_score(sub["label_true_et"], sub[score_col]))
            p = float(mannwhitneyu(
                sub[sub["label_true_et"].eq(1)][score_col],
                sub[sub["label_true_et"].eq(0)][score_col],
                alternative="two-sided",
            ).pvalue)
        true_scores = true[score_col].dropna().to_numpy(float)
        conf_scores = conf[score_col].dropna().to_numpy(float)
        rows.append({
            "comparison": name,
            "score": score_col,
            "n_true_components": int(true_scores.size),
            "n_confound_components": int(conf_scores.size),
            "true_median": float(np.median(true_scores)) if true_scores.size else np.nan,
            "true_p25": float(np.percentile(true_scores, 25)) if true_scores.size else np.nan,
            "true_p75": float(np.percentile(true_scores, 75)) if true_scores.size else np.nan,
            "confound_median": float(np.median(conf_scores)) if conf_scores.size else np.nan,
            "confound_p25": float(np.percentile(conf_scores, 25)) if conf_scores.size else np.nan,
            "confound_p75": float(np.percentile(conf_scores, 75)) if conf_scores.size else np.nan,
            "auc_true_higher_than_confound": auc,
            "mannwhitney_p": p,
            "diagnostic_verdict": (
                "persistence_separates_push_morse"
                if np.isfinite(auc) and auc >= 0.75 and np.nanmedian(conf_scores) < np.nanmedian(true_scores)
                else "persistence_overlaps_or_confound_high_wrap_up"
            ),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p3_persistence_diagnostic_summary.csv", index=False)
    return out


def labels_for_mask(mask: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    return component_labels(mask, DEFAULT_CONFIG.min_component_size_vox)


def filter_topk(pred: np.ndarray, comp: pd.DataFrame, k: int) -> np.ndarray:
    if k <= 0 or comp.empty:
        return np.zeros_like(pred, dtype=bool)
    labels, _, _ = labels_for_mask(pred)
    ranked = comp.sort_values(
        ["ph_h0_max_persistence", "volume_vox", "enh_max"],
        ascending=[False, False, False],
    )
    keep = ranked["component_label"].astype(int).head(k).tolist()
    return np.isin(labels, keep)


def score_et_case_fast(gt_et: np.ndarray, pred_et: np.ndarray) -> Dict[str, object]:
    gt = gt_et.astype(bool)
    pred = pred_et.astype(bool)
    gt_vox = int(gt.sum())
    pred_vox = int(pred.sum())
    out: Dict[str, object] = {
        "gt_vox": gt_vox,
        "pred_vox": pred_vox,
        "global_dice": np.nan if gt_vox == 0 else dice(pred, gt),
        "global_jaccard": np.nan if gt_vox == 0 else jaccard(pred, gt),
        "overseg_ratio": np.nan if gt_vox == 0 else pred_vox / gt_vox,
        "correct_absent_pred_lt_10_vox": (
            bool(pred_vox < DEFAULT_CONFIG.absent_tolerance_vox) if gt_vox == 0 else np.nan
        ),
        "flood_gt_10000_vox": bool(pred_vox > DEFAULT_CONFIG.flood_threshold_vox),
    }
    lesion = fp_aware_lesionwise_dice(gt, pred, DEFAULT_CONFIG.min_component_size_vox)
    out.update(lesion)
    if gt_vox == 0:
        for key in [
            "matched_components", "lesion_tp", "lesion_fn", "lesion_fp",
            "lesion_dice_sum", "lesionwise_dice_mean", "lesionwise_dice_median",
            "lesion_detection_rate_dice_gt_0", "lesion_detection_rate_dice_ge_0_1",
        ]:
            out[key] = np.nan
    return out


def score_case(case_id: str, vol_bin: str, method: str, k: int, components: pd.DataFrame) -> Dict[str, object]:
    seg, _ = read_image(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
    gt = as_et(seg)
    pred, _ = read_image(pred_path(case_id, method))
    pred = pred > 0
    comp = components[
        components["case_id"].eq(case_id)
        & components["method"].eq(method)
        & components["source"].isin(["prediction_component", "absent_fp_component"])
    ].copy()
    filtered = filter_topk(pred, comp, k)
    metrics = score_et_case_fast(gt, filtered)
    return {
        "case_id": case_id,
        "method": method,
        "vol_bin": vol_bin,
        "k_top_persistent_components": int(k),
        **metrics,
    }


def score_all_k(process: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    out_path = OUT / "p3_count_rank_all_k_case_metrics.csv"
    if out_path.exists():
        return pd.read_csv(out_path)
    cache_dir = OUT / "count_rank_all_k_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, meta in process.reset_index(drop=True).iterrows():
        case_id = str(meta["case_id"])
        cache_path = cache_dir / f"{case_id}.csv"
        if cache_path.exists():
            print(f"[P3 score] {idx + 1:03d}/{len(process)} {case_id} [cache]", flush=True)
            rows.extend(pd.read_csv(cache_path).to_dict("records"))
            continue
        vol_bin = str(meta["vol_bin"])
        fold = int(meta["fold"])
        print(f"[P3 score] {idx + 1:03d}/{len(process)} {case_id}", flush=True)
        case_rows = []
        seg, _ = read_image(CLEAN_ROOT / case_id / f"{case_id}-seg.nii.gz")
        gt = as_et(seg)
        for method in METHODS:
            pred, _ = read_image(pred_path(case_id, method))
            pred = pred > 0
            comp = components[
                components["case_id"].eq(case_id)
                & components["method"].eq(method)
                & components["source"].isin(["prediction_component", "absent_fp_component"])
            ].copy()
            for k in range(0, MAX_K + 1):
                filtered = filter_topk(pred, comp, k)
                metrics = score_et_case_fast(gt, filtered)
                case_rows.append({
                    "case_id": case_id,
                    "fold": fold,
                    "method": method,
                    "vol_bin": vol_bin,
                    "k_top_persistent_components": int(k),
                    **metrics,
                })
        pd.DataFrame(case_rows).to_csv(cache_path, index=False)
        rows.extend(case_rows)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def choose_k_by_train_fold(process: pd.DataFrame, all_k_scores: pd.DataFrame) -> pd.DataFrame:
    stage4 = pd.read_csv(STAGE4_CASE_METRICS).rename(columns={"metodo": "method"})
    rows = []
    for fold in sorted(process["fold"].astype(int).unique()):
        train = process[process["fold"].astype(int) != fold].copy()
        for method in METHODS:
            base_large = stage4[
                stage4["case_id"].isin(train["case_id"])
                & stage4["method"].eq(method)
                & stage4["vol_bin"].eq("large")
            ]["lesionwise_dice_mean"].mean()
            candidates = []
            for k in range(0, MAX_K + 1):
                scored = all_k_scores[
                    all_k_scores["case_id"].isin(train["case_id"])
                    & all_k_scores["method"].eq(method)
                    & all_k_scores["k_top_persistent_components"].astype(int).eq(k)
                ].copy()
                absent = scored[scored["vol_bin"].eq("absent")]
                large = scored[scored["vol_bin"].eq("large")]
                large_mean = float(large["lesionwise_dice_mean"].mean()) if len(large) else np.nan
                retention = large_mean / base_large if base_large and np.isfinite(base_large) else np.nan
                candidates.append({
                    "holdout_fold": fold,
                    "method": method,
                    "k": k,
                    "train_absent_flood_rate": float(absent["flood_gt_10000_vox"].mean()) if len(absent) else np.nan,
                    "train_absent_median_fp_vox": float(absent["pred_vox"].median()) if len(absent) else np.nan,
                    "train_large_lesionwise_mean": large_mean,
                    "train_large_own_baseline_mean": float(base_large),
                    "train_large_retention_vs_own_baseline": float(retention),
                    "train_keeps_90pct_large": bool(np.isfinite(retention) and retention >= 0.90),
                })
            cand = pd.DataFrame(candidates)
            feasible = cand[cand["train_keeps_90pct_large"]].copy()
            if feasible.empty:
                selected = cand.iloc[(cand["train_large_retention_vs_own_baseline"] - 0.90).abs().argsort().iloc[0]].to_dict()
                selected["selection_policy"] = "closest_to_90pct_large_no_feasible_k"
            else:
                selected = feasible.sort_values(
                    ["train_absent_flood_rate", "train_absent_median_fp_vox", "k"],
                    ascending=[True, True, True],
                ).iloc[0].to_dict()
                selected["selection_policy"] = "min_flood_then_median_fp_subject_to_train_large_ge_90pct"
            rows.extend(candidates)
            rows[-len(candidates)]["selected_k_for_fold_method"] = np.nan
            selected_row = selected.copy()
            selected_row["selected_k_for_fold_method"] = int(selected["k"])
            selected_row["is_selected"] = True
            rows.append(selected_row)
    out = pd.DataFrame(rows)
    if "is_selected" not in out:
        out["is_selected"] = False
    out["is_selected"] = out["is_selected"].fillna(False).astype(bool)
    out.to_csv(OUT / "p3_count_rank_thresholds_by_fold.csv", index=False)
    return out


def evaluate_count_prior(process: pd.DataFrame,
                         all_k_scores: pd.DataFrame,
                         thresholds: pd.DataFrame) -> pd.DataFrame:
    out_path = OUT / "p3_count_rank_case_metrics.csv"
    rows = []
    selected = thresholds[thresholds["is_selected"]].copy()
    for _, meta in process.iterrows():
        case_id = str(meta["case_id"])
        fold = int(meta["fold"])
        vol_bin = str(meta["vol_bin"])
        for method in METHODS:
            row = selected[selected["holdout_fold"].astype(int).eq(fold) & selected["method"].eq(method)].iloc[0]
            k = int(row["selected_k_for_fold_method"])
            scored = all_k_scores[
                all_k_scores["case_id"].eq(case_id)
                & all_k_scores["method"].eq(method)
                & all_k_scores["k_top_persistent_components"].astype(int).eq(k)
            ].iloc[0].to_dict()
            scored["vol_bin"] = vol_bin
            scored["selection_policy"] = row["selection_policy"]
            rows.append(scored)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def aggregate(case_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in case_metrics.groupby("method"):
        absent = sub[sub["vol_bin"].eq("absent")]
        row = {
            "method": method,
            "construction_type": METHOD_TYPES[method],
            "n_cases": int(sub["case_id"].nunique()),
            "absent_n": int(len(absent)),
            "absent_flood_rate": float(absent["flood_gt_10000_vox"].mean()),
            "absent_median_fp_vox": float(absent["pred_vox"].median()),
            "absent_max_fp_vox": int(absent["pred_vox"].max()) if len(absent) else 0,
            "present_dominant_found_rate": float(sub[sub["vol_bin"].isin(VOL_ORDER)]["lesion_tp"].fillna(0).gt(0).mean()),
        }
        for vol in VOL_ORDER:
            s = sub[sub["vol_bin"].eq(vol)]
            row[f"{vol}_n"] = int(len(s))
            row[f"{vol}_lesionwise_mean"] = float(s["lesionwise_dice_mean"].mean()) if len(s) else np.nan
            row[f"{vol}_dominant_found_rate"] = float(s["lesion_tp"].fillna(0).gt(0).mean()) if len(s) else np.nan
            row[f"{vol}_lesion_tp"] = int(s["lesion_tp"].sum(skipna=True)) if len(s) else 0
            row[f"{vol}_lesion_fn"] = int(s["lesion_fn"].sum(skipna=True)) if len(s) else 0
            row[f"{vol}_lesion_fp"] = int(s["lesion_fp"].sum(skipna=True)) if len(s) else 0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p3_count_rank_summary.csv", index=False)
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


def key_comparison(case_metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    stage4 = pd.read_csv(STAGE4_CASE_METRICS)
    otsu_abs = stage4[(stage4["metodo"].eq("otsu_T1c")) & (stage4["vol_bin"].eq("absent"))]
    otsu_abs = otsu_abs[["case_id", "pred_vox", "flood_gt_10000_vox"]].rename(
        columns={"pred_vox": "otsu_pred_vox", "flood_gt_10000_vox": "otsu_flood"})
    det_base = detection_baseline_by_case()
    rows = []
    for _, base in summary.iterrows():
        method = str(base["method"])
        sub = case_metrics[case_metrics["method"].eq(method)].copy()
        absent = sub[sub["vol_bin"].eq("absent")].merge(otsu_abs, on="case_id", how="inner")
        flood_delta = absent["flood_gt_10000_vox"].astype(float).to_numpy() - absent["otsu_flood"].astype(float).to_numpy()
        fp_delta = absent["pred_vox"].astype(float).to_numpy() - absent["otsu_pred_vox"].astype(float).to_numpy()
        row = base.to_dict()
        row["delta_absent_flood_vs_otsu"] = float(np.mean(flood_delta))
        row["delta_absent_flood_ci_low"], row["delta_absent_flood_ci_high"] = bootstrap_ci(flood_delta)
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
            row[f"{vol}_detection_baseline_mean"] = float(pv["detection_baseline_lesionwise"].mean()) if len(pv) else np.nan
            row[f"{vol}_retention_vs_detection_baseline"] = (
                row[f"{vol}_lesionwise_mean"] / row[f"{vol}_detection_baseline_mean"]
                if row[f"{vol}_detection_baseline_mean"] else np.nan
            )
            row[f"{vol}_delta_vs_detection_best"] = float(np.nanmean(delta)) if len(delta) else np.nan
            row[f"{vol}_delta_ci_low"], row[f"{vol}_delta_ci_high"] = bootstrap_ci(delta)
            row[f"{vol}_baseline_dominant_found_rate"] = float(pv["detection_baseline_dominant_found"].mean()) if len(pv) else np.nan
            row[f"{vol}_dominant_found_delta"] = row[f"{vol}_dominant_found_rate"] - row[f"{vol}_baseline_dominant_found_rate"]
        row["large_detection_preserved_90pct"] = bool(row["large_retention_vs_detection_baseline"] >= 0.90)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p3_key_comparison_vs_baseline.csv", index=False)
    return out


def p2b_comparison(p3_key: pd.DataFrame) -> pd.DataFrame:
    p2b = pd.read_csv(P2B_OPS)
    selected = p2b[p2b["selection"].eq("global_best_fp_while_large_detection_ge_90pct")]
    if selected.empty:
        selected = p2b[p2b["keeps_large_detection_90pct"]].sort_values(["absent_flood_rate", "absent_median_fp_vox"]).head(1)
    rows = []
    if len(selected):
        row = selected.iloc[0]
        rows.append({
            "prior": "P2b soft shape",
            "method": row["method"],
            "absent_flood_rate": row["absent_flood_rate"],
            "absent_median_fp_vox": row["absent_median_fp_vox"],
            "large_lesionwise": row["large_lesionwise_mean"],
            "large_detection_ratio": row["large_detection_ratio"],
            "large_detection_preserved_90pct": row["keeps_large_detection_90pct"],
            "selection": row["selection"],
        })
    viable = p3_key[p3_key["large_detection_preserved_90pct"]].copy()
    if viable.empty:
        best = p3_key.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
        selection = "best_fp_no_90pct_detection_candidate"
    else:
        best = viable.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
        selection = "best_fp_with_large_detection_ge_90pct"
    rows.append({
        "prior": "P3 cubical H0 persistence count/rank",
        "method": best["method"],
        "absent_flood_rate": best["absent_flood_rate"],
        "absent_median_fp_vox": best["absent_median_fp_vox"],
        "large_lesionwise": best["large_lesionwise_mean"],
        "large_detection_ratio": best["large_retention_vs_detection_baseline"],
        "large_detection_preserved_90pct": best["large_detection_preserved_90pct"],
        "selection": selection,
    })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p3_vs_p2b_comparison.csv", index=False)
    return out


def type_interaction(p3_key: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for typ, sub in p3_key.groupby("construction_type"):
        rows.append({
            "construction_type": typ,
            "n_methods": len(sub),
            "mean_absent_flood_rate": float(sub["absent_flood_rate"].mean()),
            "best_absent_flood_rate": float(sub["absent_flood_rate"].min()),
            "mean_absent_median_fp_vox": float(sub["absent_median_fp_vox"].mean()),
            "mean_large_retention_vs_detection_baseline": float(sub["large_retention_vs_detection_baseline"].mean()),
            "n_methods_preserve_large_90pct": int(sub["large_detection_preserved_90pct"].sum()),
            "best_method_by_fp_with_90pct_gate": (
                sub[sub["large_detection_preserved_90pct"]]
                .sort_values(["absent_flood_rate", "absent_median_fp_vox"])["method"]
                .iloc[0]
                if sub["large_detection_preserved_90pct"].any() else ""
            ),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p3_type_interaction.csv", index=False)
    return out


def persistence_gap_summary(components: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred = components[components["source"].isin(["prediction_component", "absent_fp_component"])].copy()
    rows = []
    for (case_id, method), sub in pred.groupby(["case_id", "method"]):
        scores = np.sort(pd.to_numeric(
            sub["ph_h0_max_persistence"], errors="coerce").dropna().to_numpy())[::-1]
        if scores.size == 0:
            gap_k = np.nan
            max_gap = np.nan
        elif scores.size == 1:
            gap_k = 1
            max_gap = np.nan
        else:
            gaps = scores[:-1] - scores[1:]
            gap_k = int(np.argmax(gaps) + 1)
            max_gap = float(np.max(gaps))
        rows.append({
            "case_id": case_id,
            "method": method,
            "n_scored_components": int(scores.size),
            "top_score": float(scores[0]) if scores.size else np.nan,
            "second_score": float(scores[1]) if scores.size > 1 else np.nan,
            "largest_gap_k": gap_k,
            "largest_gap": max_gap,
        })
    case_gap = pd.DataFrame(rows)
    case_gap.to_csv(OUT / "p3_persistence_gap_by_case_method.csv", index=False)
    summary_rows = []
    for method, sub in case_gap.groupby("method"):
        summary_rows.append({
            "method": method,
            "n_case_methods": int(len(sub)),
            "median_largest_gap_k": float(sub["largest_gap_k"].median()),
            "frac_gap_k_eq_1": float(sub["largest_gap_k"].eq(1).mean()),
            "median_largest_gap": float(sub["largest_gap"].median(skipna=True)),
            "median_top_score": float(sub["top_score"].median(skipna=True)),
            "median_second_score": float(sub["second_score"].median(skipna=True)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "p3_persistence_gap_summary.csv", index=False)
    return case_gap, summary


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


def write_report(diag: pd.DataFrame,
                 thresholds: pd.DataFrame,
                 key: pd.DataFrame,
                 p2b: pd.DataFrame,
                 types: pd.DataFrame,
                 gap_summary: pd.DataFrame) -> None:
    verdict = diag[diag["comparison"].eq("all_confound")]["diagnostic_verdict"].iloc[0]
    path = OUT / "p3_morse_diagnostic_report.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# P3 Cubical Persistence / Morse Diagnostic\n\n")
        f.write("This is the first genuine topological computation in the project: ")
        f.write("GUDHI `CubicalComplex` is run on the normalized enhancement map ")
        f.write("(`max(T1c - T1n, 0)` scaled by its positive p99.5). ")
        f.write("For each connected component, PH is computed on the component crop with outside-component voxels set to zero. ")
        f.write("Superlevel-set H0 persistence is obtained by running sublevel PH on `-enhancement`; ")
        f.write("the essential H0 class is capped at background, giving a normalized peak-prominence score in `[0,1]`.\n\n")
        f.write("Prediction masks with many components use the diagnostic candidate policy: ")
        f.write(f"top {CANDIDATE_BY_VOLUME} by volume union top {CANDIDATE_BY_MAX_ENH} by enhancement. ")
        f.write("Components outside that candidate set are rejected by the count/rank prior. ")
        f.write("Large crops above the exact voxel budget are deterministically downsampled before PH; the downsample step is recorded per component.\n\n")
        f.write("## Diagnostic Verdict\n\n")
        f.write(f"Verdict: **{verdict}**.\n\n")
        f.write(df_to_markdown(diag))
        f.write("\n\n")
        f.write("Interpretation rule: if confound components have lower H0 persistence than true ET ")
        f.write("(AUC >= 0.75 for true > confound), persistence supports pushing toward Morse. ")
        f.write("If confounds are high-persistence or overlap true ET, intensity persistence will inherit the confound and Morse should be treated as unlikely to solve the core FP problem.\n\n")
        f.write("## Train-Fold Count/Rank Prior\n\n")
        f.write("The raw persistence-gap heuristic was not a useful standalone count rule. ")
        f.write("The brightest scored component is usually maximally persistent (`top_score` median 1.0), ")
        f.write("and the gap either collapses to `k=1` or points late in the candidate list for flood-prone methods. ")
        f.write("Therefore the applied prior uses train-fold selection of `k` rather than choosing `k` on held-out cases.\n\n")
        f.write(df_to_markdown(gap_summary))
        f.write("\n\n")
        selected = thresholds[thresholds["is_selected"]].copy()
        f.write("For each held-out fold and method, `k` was selected on the other four folds only: ")
        f.write("minimize absent flood, then median FP, subject to preserving >=90% of that method's train-fold large-lesion baseline where feasible.\n\n")
        f.write(selected[[
            "holdout_fold", "method", "selected_k_for_fold_method",
            "train_absent_flood_rate", "train_absent_median_fp_vox",
            "train_large_retention_vs_own_baseline", "selection_policy",
        ]].pipe(df_to_markdown))
        f.write("\n\n")
        f.write("## Held-Out Per-Axis Comparison\n\n")
        cols = [
            "method", "construction_type", "absent_flood_rate", "absent_median_fp_vox",
            "delta_absent_flood_vs_otsu", "delta_absent_flood_ci_low", "delta_absent_flood_ci_high",
            "large_lesionwise_mean", "large_retention_vs_detection_baseline",
            "large_detection_preserved_90pct",
        ]
        f.write(df_to_markdown(key[cols]))
        f.write("\n\n")
        f.write("## P3 vs P2b\n\n")
        f.write(df_to_markdown(p2b))
        f.write("\n\n")
        f.write("## Method-Type Interaction\n\n")
        f.write(df_to_markdown(types))
        f.write("\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    process = pd.read_csv(MANIFEST)
    process = process[process["process"].astype(int).eq(1)].copy()

    components = build_component_persistence(process)
    diag = persistence_diagnostic(components)
    _, gap_summary = persistence_gap_summary(components)
    all_k_scores = score_all_k(process, components)
    thresholds = choose_k_by_train_fold(process, all_k_scores)
    case_metrics = evaluate_count_prior(process, all_k_scores, thresholds)
    summary = aggregate(case_metrics)
    key = key_comparison(case_metrics, summary)
    p2b = p2b_comparison(key)
    types = type_interaction(key)
    write_report(diag, thresholds, key, p2b, types, gap_summary)

    print("\nPERSISTENCE DIAGNOSTIC")
    print(diag.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nP3 KEY COMPARISON")
    cols = [
        "method", "construction_type", "absent_flood_rate", "absent_median_fp_vox",
        "large_lesionwise_mean", "large_retention_vs_detection_baseline",
        "large_detection_preserved_90pct",
    ]
    print(key[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nP3 VS P2B")
    print(p2b.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nReport: {OUT / 'p3_morse_diagnostic_report.md'}")


if __name__ == "__main__":
    main()
