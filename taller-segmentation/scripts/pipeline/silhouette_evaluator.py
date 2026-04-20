"""SilhouetteEvaluator — stratified edge-aware silhouette on ROI voxels."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel
from sklearn.metrics import calinski_harabasz_score, silhouette_score


class SilhouetteEvaluator:
    """Compute the silhouette coefficient on a stratified sample of ROI voxels.

    Full-volume silhouette is intractable on 3D MR (10^6-10^7 voxels).
    We draw two sub-samples per cluster and fuse them:

        * Base sample: uniform random voxels within the cluster.
        * Edge sample: voxels with high intensity gradient magnitude
          (computed via 3D Sobel), where boundary ambiguity matters most.

    To stabilise the estimate we repeat the stratified sampling ``n_repeats``
    times and return the mean and variance.
    """

    def __init__(
        self,
        sample_size: int = 20000,
        edge_fraction: float = 0.5,
        n_repeats: int = 3,
        random_state: int = 42,
    ) -> None:
        if sample_size < 200:
            raise ValueError("sample_size too small")
        if not (0.0 <= edge_fraction <= 0.95):
            raise ValueError("edge_fraction must be in [0, 0.95]")
        self.sample_size = int(sample_size)
        self.edge_fraction = float(edge_fraction)
        self.n_repeats = int(n_repeats)
        self.random_state = int(random_state)

    def evaluate(
        self,
        intensity_array: np.ndarray,
        label_array: np.ndarray,
        roi_mask: np.ndarray,
    ) -> dict:
        if intensity_array.shape != label_array.shape != roi_mask.shape:
            raise ValueError("shape mismatch between intensity, labels, mask")

        gradient_magnitude = self._gradient_magnitude(intensity_array)

        roi = roi_mask.astype(bool)
        labels_flat = label_array[roi]
        values_flat = intensity_array[roi]
        grads_flat = gradient_magnitude[roi]

        uniq = np.unique(labels_flat)
        if uniq.size < 2:
            return {
                "score": float("nan"),
                "variance": float("nan"),
                "calinski_harabasz": float("nan"),
                "calinski_harabasz_variance": float("nan"),
                "n_clusters_sampled": int(uniq.size),
            }

        sil_scores: list[float] = []
        ch_scores: list[float] = []
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_repeats):
            idx = self._stratified_indices(labels_flat, grads_flat, uniq, rng)
            if idx.size < 2 * uniq.size:
                continue
            sample_values = values_flat[idx].reshape(-1, 1)
            sample_labels = labels_flat[idx]
            if np.unique(sample_labels).size < 2:
                continue
            sil_scores.append(
                float(silhouette_score(sample_values, sample_labels, metric="euclidean"))
            )
            ch_scores.append(
                float(calinski_harabasz_score(sample_values, sample_labels))
            )

        if not sil_scores:
            return {
                "score": float("nan"),
                "variance": float("nan"),
                "calinski_harabasz": float("nan"),
                "calinski_harabasz_variance": float("nan"),
                "n_clusters_sampled": int(uniq.size),
            }

        return {
            "score": float(np.mean(sil_scores)),
            "variance": float(np.var(sil_scores)),
            "calinski_harabasz": float(np.mean(ch_scores)),
            "calinski_harabasz_variance": float(np.var(ch_scores)),
            "n_repeats": len(sil_scores),
            "n_clusters_sampled": int(uniq.size),
            "sample_size": int(self.sample_size),
        }

    def _stratified_indices(
        self,
        labels: np.ndarray,
        gradients: np.ndarray,
        unique_labels: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        per_cluster = max(50, self.sample_size // unique_labels.size)
        n_edge = int(per_cluster * self.edge_fraction)
        n_base = per_cluster - n_edge

        chosen: list[np.ndarray] = []
        for lbl in unique_labels:
            cluster_idx = np.flatnonzero(labels == lbl)
            if cluster_idx.size == 0:
                continue
            if cluster_idx.size <= per_cluster:
                chosen.append(cluster_idx)
                continue
            cluster_grads = gradients[cluster_idx]
            edge_cut = np.percentile(cluster_grads, 75)
            edge_pool = cluster_idx[cluster_grads >= edge_cut]
            base_pool = cluster_idx[cluster_grads < edge_cut]
            if edge_pool.size == 0:
                edge_pool = cluster_idx
            if base_pool.size == 0:
                base_pool = cluster_idx

            edge_take = min(n_edge, edge_pool.size)
            base_take = min(n_base, base_pool.size)
            chosen.append(rng.choice(edge_pool, size=edge_take, replace=False))
            chosen.append(rng.choice(base_pool, size=base_take, replace=False))

        return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)

    @staticmethod
    def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
        gz = sobel(array, axis=0, mode="nearest")
        gy = sobel(array, axis=1, mode="nearest")
        gx = sobel(array, axis=2, mode="nearest")
        return np.sqrt(gz * gz + gy * gy + gx * gx)
