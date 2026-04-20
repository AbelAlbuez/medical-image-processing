"""IntensityPreprocessor — shift, N4 bias correction, ROI-scoped normalization."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from .volume_io import VolumeIO


class IntensityPreprocessor:
    """Robust MR intensity preparation restricted to an anatomical ROI.

    Steps
    -----
    1. Cast to float32.
    2. Shift so that ``min >= min_positive`` (N4 uses log internally).
    3. Run N4 bias field correction with the ROI as mask.
    4. Normalize *within the ROI only* (z-score or min-max) and force
       out-of-ROI voxels to exactly 0 so they cannot distort K-means.
    """

    def __init__(
        self,
        n4_iterations: tuple[int, ...] = (50, 50, 30, 20),
        n4_shrink: int = 4,
        normalization: str = "zscore",
        min_positive: float = 1.0,
    ) -> None:
        if normalization not in {"zscore", "minmax"}:
            raise ValueError("normalization must be 'zscore' or 'minmax'")
        self.n4_iterations = list(int(x) for x in n4_iterations)
        self.n4_shrink = int(n4_shrink)
        self.normalization = normalization
        self.min_positive = float(min_positive)

    def run(self, image: sitk.Image, roi_mask: sitk.Image) -> tuple[sitk.Image, dict]:
        arr = VolumeIO.to_numpy(image).astype(np.float32)
        roi = VolumeIO.to_numpy(roi_mask).astype(bool)
        if not roi.any():
            raise RuntimeError("ROI is empty — cannot preprocess intensities")

        shift = 0.0
        roi_min = float(arr[roi].min())
        if roi_min < self.min_positive:
            shift = self.min_positive - roi_min
            arr = arr + shift
        shifted_img = VolumeIO.from_numpy(arr, image, pixel_type=sitk.sitkFloat32)

        mask_img = sitk.Cast(roi_mask, sitk.sitkUInt8)

        n4 = sitk.N4BiasFieldCorrectionImageFilter()
        n4.SetMaximumNumberOfIterations(self.n4_iterations)
        n4.SetMaskLabel(1)
        if self.n4_shrink > 1:
            shrunk_img = sitk.Shrink(shifted_img, [self.n4_shrink] * 3)
            shrunk_mask = sitk.Shrink(mask_img, [self.n4_shrink] * 3)
            n4.Execute(shrunk_img, shrunk_mask)
            log_bias = n4.GetLogBiasFieldAsImage(shifted_img)
            corrected = shifted_img / sitk.Exp(log_bias)
            corrected = sitk.Cast(corrected, sitk.sitkFloat32)
        else:
            corrected = n4.Execute(shifted_img, mask_img)
            corrected = sitk.Cast(corrected, sitk.sitkFloat32)
        corrected.CopyInformation(image)

        corrected_arr = VolumeIO.to_numpy(corrected)
        roi_values = corrected_arr[roi]

        if self.normalization == "zscore":
            mu = float(roi_values.mean())
            sigma = float(roi_values.std())
            if sigma < 1e-6:
                sigma = 1.0
            normalized = (corrected_arr - mu) / sigma
            stats = {"mode": "zscore", "mean": mu, "std": sigma}
        else:
            lo = float(np.percentile(roi_values, 1.0))
            hi = float(np.percentile(roi_values, 99.0))
            span = max(hi - lo, 1e-6)
            normalized = (corrected_arr - lo) / span
            normalized = np.clip(normalized, 0.0, 1.0)
            stats = {"mode": "minmax", "p01": lo, "p99": hi}

        normalized[~roi] = 0.0
        normalized_img = VolumeIO.from_numpy(
            normalized.astype(np.float32), image, pixel_type=sitk.sitkFloat32
        )

        stats.update(
            {
                "shift_applied": shift,
                "roi_voxels": int(roi.sum()),
                "post_roi_min": float(normalized[roi].min()),
                "post_roi_max": float(normalized[roi].max()),
                "post_roi_mean": float(normalized[roi].mean()),
            }
        )
        return normalized_img, stats
