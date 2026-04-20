"""ForegroundMaskBuilder — Otsu/Li/percentile + morphology for anatomical ROI."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from .volume_io import VolumeIO


class ForegroundMaskBuilder:
    """Build a binary anatomical ROI and eliminate air/background.

    Parameters
    ----------
    threshold_method : {"otsu", "li", "huang", "percentile"}
        Intensity thresholding method. For breast MR with diffuse fatty
        signal, ``li`` or a low ``percentile`` captures the tissue better
        than Otsu.
    percentile : float, optional
        Used only when ``threshold_method == "percentile"``. Voxels above
        this percentile of the non-zero histogram are kept.
    keep_largest : bool
        If True keep only the largest connected component. For bilateral
        organs (e.g. breast), set False and use ``min_component_voxels``.
    min_component_voxels : int
        When ``keep_largest`` is False, components smaller than this are
        discarded.
    fill_holes : bool
        Fill interior holes of the ROI (useful when anatomy has dark
        vessels or cavities that Otsu would exclude).
    """

    def __init__(
        self,
        threshold_method: str = "otsu",
        percentile: float | None = None,
        closing_radius: int = 3,
        opening_radius: int = 1,
        keep_largest: bool = True,
        min_component_voxels: int = 2000,
        fill_holes: bool = True,
        min_voxels: int = 5000,
        inner_erode_radius: int = 0,
    ) -> None:
        if threshold_method not in {"otsu", "li", "huang", "percentile"}:
            raise ValueError(f"Unknown threshold_method: {threshold_method}")
        if threshold_method == "percentile" and percentile is None:
            raise ValueError("percentile must be provided when threshold_method='percentile'")
        self.threshold_method = threshold_method
        self.percentile = percentile
        self.closing_radius = int(closing_radius)
        self.opening_radius = int(opening_radius)
        self.keep_largest = bool(keep_largest)
        self.min_component_voxels = int(min_component_voxels)
        self.fill_holes = bool(fill_holes)
        self.min_voxels = int(min_voxels)
        self.inner_erode_radius = int(inner_erode_radius)

    def build(self, image: sitk.Image) -> sitk.Image:
        mask = self._threshold(image)

        if self.closing_radius > 0:
            mask = sitk.BinaryMorphologicalClosing(
                mask, [self.closing_radius] * 3, sitk.sitkBall
            )
        if self.opening_radius > 0:
            mask = sitk.BinaryMorphologicalOpening(
                mask, [self.opening_radius] * 3, sitk.sitkBall
            )

        if self.fill_holes:
            mask = sitk.BinaryFillhole(mask)

        if self.keep_largest:
            mask = self._keep_largest_component(mask)
        elif self.min_component_voxels > 0:
            mask = self._filter_small_components(mask)

        if self.inner_erode_radius > 0:
            mask = sitk.BinaryErode(
                mask, [self.inner_erode_radius] * 3, sitk.sitkBall
            )
            if self.keep_largest:
                mask = self._keep_largest_component(mask)

        mask = sitk.Cast(mask, sitk.sitkUInt8)
        mask.CopyInformation(image)

        voxels = int(sitk.GetArrayViewFromImage(mask).sum())
        if voxels < self.min_voxels:
            raise RuntimeError(
                f"Foreground mask too small ({voxels} voxels); "
                "check threshold/morphology configuration."
            )
        return mask

    def _threshold(self, image: sitk.Image) -> sitk.Image:
        if self.threshold_method == "otsu":
            return sitk.OtsuThreshold(image, 0, 1)
        if self.threshold_method == "li":
            return sitk.LiThreshold(image, 0, 1)
        if self.threshold_method == "huang":
            return sitk.HuangThreshold(image, 0, 1)
        arr = sitk.GetArrayViewFromImage(image)
        positive = arr[arr > 0]
        if positive.size == 0:
            raise RuntimeError("Image has no positive voxels for percentile threshold")
        thresh = float(np.percentile(positive, self.percentile))
        hi = float(arr.max()) + 1.0
        return sitk.BinaryThreshold(image, thresh, hi, 1, 0)

    @staticmethod
    def _keep_largest_component(mask: sitk.Image) -> sitk.Image:
        cc = sitk.ConnectedComponent(mask)
        relabelled = sitk.RelabelComponent(cc, sortByObjectSize=True)
        return sitk.BinaryThreshold(relabelled, 1, 1, 1, 0)

    def _filter_small_components(self, mask: sitk.Image) -> sitk.Image:
        cc = sitk.ConnectedComponent(mask)
        relabelled = sitk.RelabelComponent(
            cc, minimumObjectSize=self.min_component_voxels, sortByObjectSize=True
        )
        return sitk.BinaryThreshold(relabelled, 1, 1_000_000, 1, 0)

    @staticmethod
    def bounding_box(mask: sitk.Image) -> tuple[tuple[int, int], ...]:
        arr = VolumeIO.to_numpy(mask)
        coords = np.argwhere(arr > 0)
        if coords.size == 0:
            raise RuntimeError("Empty foreground mask")
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        return tuple((int(a), int(b)) for a, b in zip(mins, maxs))
