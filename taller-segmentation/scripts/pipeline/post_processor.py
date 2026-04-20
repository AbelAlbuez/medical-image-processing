"""PostProcessor — clean the final binary mask (remove islands, close holes)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    generate_binary_structure,
    label as cc_label,
)


class PostProcessor:
    """Remove spurious components and smooth the lesion mask.

    Operations in order:
        1. Keep only connected components larger than ``min_component_voxels``.
        2. Optionally keep only the top ``max_components`` by size.
        3. Binary closing with a small 3D ball to bridge pinhole gaps.
        4. Fill interior holes component-by-component.
    """

    def __init__(
        self,
        min_component_voxels: int = 50,
        max_components: int | None = 3,
        closing_iterations: int = 1,
        fill_holes: bool = True,
    ) -> None:
        self.min_component_voxels = int(min_component_voxels)
        self.max_components = max_components if max_components is None else int(max_components)
        self.closing_iterations = int(closing_iterations)
        self.fill_holes = bool(fill_holes)

    def run(self, mask: np.ndarray) -> np.ndarray:
        binary = mask.astype(bool)
        if not binary.any():
            return binary.astype(np.uint8)

        structure = generate_binary_structure(3, 1)
        labelled, n = cc_label(binary, structure=structure)
        if n == 0:
            return binary.astype(np.uint8)

        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        keep = np.zeros(sizes.size, dtype=bool)
        keep[sizes >= self.min_component_voxels] = True

        if self.max_components is not None:
            order = np.argsort(sizes)[::-1]
            kept = 0
            limited = np.zeros_like(keep)
            for idx in order:
                if idx == 0 or not keep[idx]:
                    continue
                limited[idx] = True
                kept += 1
                if kept >= self.max_components:
                    break
            keep = limited

        cleaned = keep[labelled]

        if self.closing_iterations > 0 and cleaned.any():
            cleaned = binary_closing(cleaned, structure=structure, iterations=self.closing_iterations)

        if self.fill_holes and cleaned.any():
            cleaned = binary_fill_holes(cleaned)

        return cleaned.astype(np.uint8)
