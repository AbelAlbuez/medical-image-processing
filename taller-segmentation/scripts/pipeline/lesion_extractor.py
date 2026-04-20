"""LesionMaskExtractor — plausibility-driven selection of the lesion cluster(s)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_dilation


@dataclass
class ClusterScore:
    cluster_id: int
    size: int
    mean_intensity: float
    tail_extremeness: float
    compactness: float
    edge_support: float
    score: float


class LesionMaskExtractor:
    """Select the cluster (or adjacent pair) most likely to be a lesion.

    Scoring components (all normalized to [0, 1]):
        * intensity_extremeness: distance of the cluster mean from the ROI
          centre in robust z-units (using p50/IQR). Lesions often sit at
          a tail but we do not assume which tail.
        * compactness: largest-component voxels / bounding-box volume, a
          proxy for "one coherent blob" vs "scattered texture".
        * edge_support: mean gradient magnitude along the cluster boundary,
          relative to the ROI gradient mean. Real lesions have crisp edges.
        * size_penalty: penalizes clusters that are too small or dominate
          the ROI (likely healthy parenchyma).
    """

    def __init__(
        self,
        min_cluster_fraction: float = 0.0005,
        max_cluster_fraction: float = 0.10,
        consider_pairs: bool = True,
        prefer_tail: str = "auto",
    ) -> None:
        if prefer_tail not in {"auto", "bright", "dark"}:
            raise ValueError("prefer_tail must be 'auto', 'bright', or 'dark'")
        self.min_cluster_fraction = float(min_cluster_fraction)
        self.max_cluster_fraction = float(max_cluster_fraction)
        self.consider_pairs = bool(consider_pairs)
        self.prefer_tail = prefer_tail

    def extract(
        self,
        intensity_array: np.ndarray,
        label_array: np.ndarray,
        roi_mask: np.ndarray,
        gradient_magnitude: np.ndarray | None = None,
    ) -> dict:
        roi = roi_mask.astype(bool)
        roi_total = int(roi.sum())
        if roi_total == 0:
            raise RuntimeError("ROI is empty")

        if gradient_magnitude is None:
            gradient_magnitude = self._gradient_magnitude(intensity_array)

        roi_values = intensity_array[roi]
        median = float(np.median(roi_values))
        p10, p90 = np.percentile(roi_values, [10, 90])
        q25, q75 = np.percentile(roi_values, [25, 75])
        iqr = float(max(q75 - q25, 1e-6))
        roi_grad_mean = float(gradient_magnitude[roi].mean() + 1e-8)

        unique_ids = [int(x) for x in np.unique(label_array[roi])]
        unique_ids = [c for c in unique_ids if c >= 0]

        cluster_means = {
            cid: float(intensity_array[(label_array == cid) & roi].mean())
            for cid in unique_ids
            if ((label_array == cid) & roi).any()
        }
        tail = self.prefer_tail
        if tail == "auto" and cluster_means:
            median_mean = float(np.median(list(cluster_means.values())))
            means_arr = np.asarray(list(cluster_means.values()))
            bright_gap = float(means_arr.max() - median_mean)
            dark_gap = float(median_mean - means_arr.min())
            tail = "bright" if bright_gap >= dark_gap else "dark"

        scores: dict[int, ClusterScore] = {}
        for cid in unique_ids:
            mask = (label_array == cid) & roi
            size = int(mask.sum())
            frac = size / roi_total
            if frac < self.min_cluster_fraction or frac > self.max_cluster_fraction:
                continue
            mean_int = cluster_means[cid]
            if tail == "bright":
                tail_ext = max(0.0, (mean_int - p90) / iqr)
            else:
                tail_ext = max(0.0, (p10 - mean_int) / iqr)
            tail_ext = float(min(tail_ext, 1.0))
            compactness = self._compactness(mask)
            edge = self._edge_support(mask, gradient_magnitude, roi_grad_mean)
            score = 0.5 * tail_ext + 0.25 * compactness + 0.25 * edge
            scores[cid] = ClusterScore(
                cluster_id=cid,
                size=size,
                mean_intensity=mean_int,
                tail_extremeness=tail_ext,
                compactness=compactness,
                edge_support=edge,
                score=score,
            )

        if not scores:
            if tail == "bright":
                fallback = max(unique_ids, key=lambda c: cluster_means.get(c, -np.inf))
            else:
                fallback = min(unique_ids, key=lambda c: cluster_means.get(c, np.inf))
            mask = (label_array == fallback) & roi
            return {
                "chosen_ids": [int(fallback)],
                "binary_mask": mask.astype(np.uint8),
                "scores": {},
                "reason": "fallback_no_eligible_cluster",
                "tail": tail,
            }

        best_single = max(scores.values(), key=lambda s: s.score)
        chosen_ids = [best_single.cluster_id]
        best_score = best_single.score
        reason = "best_single_cluster"

        if self.consider_pairs and len(scores) >= 2:
            sorted_by_mean = sorted(scores.keys(), key=lambda c: cluster_means[c])
            for i, a in enumerate(sorted_by_mean[:-1]):
                b = sorted_by_mean[i + 1]
                mask_pair = ((label_array == a) | (label_array == b)) & roi
                size = int(mask_pair.sum())
                frac = size / roi_total
                if frac > self.max_cluster_fraction:
                    continue
                mean_int = float(intensity_array[mask_pair].mean())
                if tail == "bright":
                    tail_ext = max(0.0, (mean_int - p90) / iqr)
                else:
                    tail_ext = max(0.0, (p10 - mean_int) / iqr)
                tail_ext = float(min(tail_ext, 1.0))
                comp = self._compactness(mask_pair)
                edge = self._edge_support(mask_pair, gradient_magnitude, roi_grad_mean)
                pair_score = 0.5 * tail_ext + 0.25 * comp + 0.25 * edge
                if pair_score > best_score + 0.05:
                    best_score = pair_score
                    chosen_ids = [a, b]
                    reason = "adjacent_pair"

        chosen_mask = np.zeros_like(label_array, dtype=bool)
        for cid in chosen_ids:
            chosen_mask |= (label_array == cid) & roi

        return {
            "chosen_ids": [int(c) for c in chosen_ids],
            "binary_mask": chosen_mask.astype(np.uint8),
            "scores": {
                int(cid): {
                    "size": s.size,
                    "mean_intensity": s.mean_intensity,
                    "tail_extremeness": s.tail_extremeness,
                    "compactness": s.compactness,
                    "edge_support": s.edge_support,
                    "score": s.score,
                }
                for cid, s in scores.items()
            },
            "combined_score": float(best_score),
            "reason": reason,
            "tail": tail,
        }

    @staticmethod
    def _compactness(mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        labelled, n = cc_label(mask)
        if n == 0:
            return 0.0
        sizes = np.bincount(labelled.ravel())[1:]
        largest_size = int(sizes.max())
        coords = np.argwhere(labelled == (int(np.argmax(sizes)) + 1))
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        bbox_volume = float(np.prod(maxs - mins + 1))
        if bbox_volume <= 0:
            return 0.0
        return float(min(largest_size / bbox_volume, 1.0))

    @staticmethod
    def _edge_support(
        mask: np.ndarray, gradient_magnitude: np.ndarray, baseline: float
    ) -> float:
        if not mask.any():
            return 0.0
        dilated = binary_dilation(mask, iterations=1)
        border = dilated & ~mask
        if not border.any():
            return 0.0
        border_mean = float(gradient_magnitude[border].mean())
        ratio = border_mean / baseline
        return float(min(ratio / 3.0, 1.0))

    @staticmethod
    def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
        from scipy.ndimage import sobel

        gz = sobel(array, axis=0, mode="nearest")
        gy = sobel(array, axis=1, mode="nearest")
        gx = sobel(array, axis=2, mode="nearest")
        return np.sqrt(gz * gz + gy * gy + gx * gx)
