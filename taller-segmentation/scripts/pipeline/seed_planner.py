"""SeedPlanner — deterministic quantile seeds with tail anchors + warm-start."""
from __future__ import annotations

import numpy as np


class SeedPlanner:
    """Generate multiple deterministic seed sets for a given K.

    Two strategies are produced per K:
        * ``quantile_anchored``: percentile-based seeds with explicit anchors
          near both intensity tails (default p2 and p98) so that hyper- and
          hypo-intense pathologies are always represented.
        * ``warm_start``: takes the final means of the best K-1 run and adds
          one new seed at the most under-represented quantile. Returns an
          empty list if no previous means are provided.
    """

    def __init__(
        self,
        low_anchor: float = 2.0,
        high_anchor: float = 98.0,
    ) -> None:
        if not (0.0 < low_anchor < high_anchor < 100.0):
            raise ValueError("Invalid anchor percentiles")
        self.low_anchor = float(low_anchor)
        self.high_anchor = float(high_anchor)

    def quantile_anchored(self, roi_values: np.ndarray, k: int) -> list[float]:
        if k < 2:
            raise ValueError("K must be >= 2")
        if k == 2:
            percentiles = [self.low_anchor, self.high_anchor]
        else:
            inner = np.linspace(self.low_anchor, self.high_anchor, k)
            percentiles = inner.tolist()
        seeds = np.percentile(roi_values, percentiles)
        seeds = np.sort(np.asarray(seeds, dtype=np.float64))
        seeds = self._dedupe(seeds)
        return seeds.tolist()

    def warm_start(
        self,
        roi_values: np.ndarray,
        previous_means: list[float] | None,
        k: int,
    ) -> list[float]:
        if previous_means is None or len(previous_means) + 1 != k:
            return []
        means = np.sort(np.asarray(previous_means, dtype=np.float64))
        gaps = np.diff(np.concatenate(([roi_values.min()], means, [roi_values.max()])))
        biggest = int(np.argmax(gaps))
        edges = np.concatenate(([roi_values.min()], means, [roi_values.max()]))
        new_seed = 0.5 * (edges[biggest] + edges[biggest + 1])
        seeds = np.sort(np.append(means, new_seed))
        seeds = self._dedupe(seeds)
        if seeds.size != k:
            return []
        return seeds.tolist()

    @staticmethod
    def _dedupe(seeds: np.ndarray, tol: float = 1e-4) -> np.ndarray:
        seeds = np.sort(seeds)
        kept = [seeds[0]]
        for s in seeds[1:]:
            if abs(s - kept[-1]) > tol:
                kept.append(s)
            else:
                kept.append(kept[-1] + tol)
        return np.asarray(kept, dtype=np.float64)

    def plan(
        self,
        roi_values: np.ndarray,
        k: int,
        previous_means: list[float] | None = None,
    ) -> dict[str, list[float]]:
        plans: dict[str, list[float]] = {
            "quantile_anchored": self.quantile_anchored(roi_values, k),
        }
        warm = self.warm_start(roi_values, previous_means, k)
        if warm:
            plans["warm_start"] = warm
        return plans
