"""Locked Stage-4 / Phase-2 ET scoring functions.

WARNING: this module is the study's locked metric surface. Phase 2 baselines,
spatial priors, shape-proxy priors, cubical-persistence diagnostics, and the
paper tables all depend on these exact definitions. Do not alter formulas,
component filtering, or absent-case thresholds without regenerating and
explicitly re-auditing every downstream result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


STRUCT26 = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class MetricConfig:
    min_component_size_vox: int = 10
    absent_tolerance_vox: int = 10
    large_fp_threshold_vox: int = 1_000
    flood_threshold_vox: int = 10_000


DEFAULT_CONFIG = MetricConfig()


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


def component_labels(mask: np.ndarray,
                     min_size: int = DEFAULT_CONFIG.min_component_size_vox
                     ) -> Tuple[np.ndarray, int, np.ndarray]:
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


def fp_aware_lesionwise_dice(
    gt: np.ndarray,
    pred: np.ndarray,
    min_component_size: int = DEFAULT_CONFIG.min_component_size_vox,
) -> Dict[str, float]:
    """Detection-aware lesion-wise Dice used for Phase 2.

    Components smaller than ``min_component_size`` are filtered in both GT and
    prediction. Positive-overlap GT/pred components are one-to-one matched by
    Hungarian assignment on Dice. The score is:

        sum(matched Dice) / (TP + FN + FP)

    where TP is the number of positive-overlap matches, FN unmatched GT
    components, and FP unmatched prediction components.
    """
    gt_lab, n_gt, gt_sizes = component_labels(gt, min_component_size)
    pred_lab, n_pred, pred_sizes = component_labels(pred, min_component_size)

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
            "lesion_fn": int(n_gt),
            "lesion_fp": 0,
            "lesionwise_dice_mean": 0.0,
            "lesionwise_dice_median": 0.0,
            "lesion_detection_rate_dice_gt_0": 0.0,
            "lesion_detection_rate_dice_ge_0_1": 0.0,
        })
        return out

    scores = np.zeros((n_gt, n_pred), dtype=np.float32)
    both = (gt_lab > 0) & (pred_lab > 0)
    if np.any(both):
        pairs = np.stack([gt_lab[both], pred_lab[both]], axis=1)
        pair_ids, counts = np.unique(pairs, axis=0, return_counts=True)
        for (gt_idx, pred_idx), inter in zip(pair_ids, counts):
            denom = gt_sizes[gt_idx] + pred_sizes[pred_idx]
            if denom > 0:
                scores[gt_idx - 1, pred_idx - 1] = 2.0 * inter / denom

    gt_match_idx, pred_match_idx = linear_sum_assignment(-scores)
    per_gt = np.zeros(n_gt, dtype=np.float32)
    per_gt[gt_match_idx] = scores[gt_match_idx, pred_match_idx]
    positive_matches = scores[gt_match_idx, pred_match_idx][
        scores[gt_match_idx, pred_match_idx] > 0
    ]

    tp = int(positive_matches.size)
    fn = int(n_gt - tp)
    fp = int(n_pred - tp)
    denom = tp + fn + fp
    lesionwise = float(positive_matches.sum() / denom) if denom else np.nan

    out.update({
        "matched_components": tp,
        "lesion_tp": tp,
        "lesion_fn": fn,
        "lesion_fp": fp,
        "lesion_dice_sum": float(positive_matches.sum()),
        "lesionwise_dice_mean": lesionwise,
        "lesionwise_dice_median": float(np.median(per_gt)),
        "lesion_detection_rate_dice_gt_0": float((per_gt > 0).mean()),
        "lesion_detection_rate_dice_ge_0_1": float((per_gt >= 0.1).mean()),
    })
    return out


def _surface(mask: np.ndarray) -> np.ndarray:
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

    gt_surface = _surface(gt)
    pred_surface = _surface(pred)
    if not gt_surface.any() or not pred_surface.any():
        return np.nan

    sampling_zyx = tuple(reversed(tuple(float(x) for x in spacing_xyz)))
    dt_gt = ndimage.distance_transform_edt(~gt_surface, sampling=sampling_zyx)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=sampling_zyx)
    distances = np.concatenate([dt_pred[gt_surface], dt_gt[pred_surface]])
    if distances.size == 0:
        return np.nan
    return float(np.percentile(distances, 95))


def score_et_case(
    gt_et: np.ndarray,
    pred_et: np.ndarray,
    spacing_xyz: Iterable[float] = (1.0, 1.0, 1.0),
    config: MetricConfig = DEFAULT_CONFIG,
) -> Dict[str, float]:
    gt = gt_et.astype(bool)
    pred = pred_et.astype(bool)
    gt_vox = int(gt.sum())
    pred_vox = int(pred.sum())

    out: Dict[str, float] = {
        "gt_vox": gt_vox,
        "pred_vox": pred_vox,
        "global_dice": np.nan if gt_vox == 0 else dice(pred, gt),
        "global_jaccard": np.nan if gt_vox == 0 else jaccard(pred, gt),
        "overseg_ratio": np.nan if gt_vox == 0 else pred_vox / gt_vox,
        "correct_absent_pred_lt_10_vox": (
            bool(pred_vox < config.absent_tolerance_vox) if gt_vox == 0 else np.nan
        ),
        "flood_gt_10000_vox": bool(pred_vox > config.flood_threshold_vox),
    }

    lesion = fp_aware_lesionwise_dice(
        gt, pred, min_component_size=config.min_component_size_vox)
    out.update(lesion)

    if gt_vox > 0:
        out["hd95_mm"] = hd95(gt, pred, spacing_xyz)
    else:
        out["matched_components"] = np.nan
        out["lesion_tp"] = np.nan
        out["lesion_fn"] = np.nan
        out["lesion_fp"] = np.nan
        out["lesion_dice_sum"] = np.nan
        out["lesionwise_dice_mean"] = np.nan
        out["lesionwise_dice_median"] = np.nan
        out["lesion_detection_rate_dice_gt_0"] = np.nan
        out["lesion_detection_rate_dice_ge_0_1"] = np.nan
        out["hd95_mm"] = np.nan
    return out
