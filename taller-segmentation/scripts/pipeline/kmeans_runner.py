"""KMeansRunner — executes ITK's ScalarImageKmeansImageFilter (via SimpleITK)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk

from .volume_io import VolumeIO


@dataclass
class KMeansResult:
    k: int
    strategy: str
    initial_means: list[float]
    final_means: list[float] = field(default_factory=list)
    label_image: sitk.Image | None = None
    label_array: np.ndarray | None = None
    roi_label_array: np.ndarray | None = None


class KMeansRunner:
    """Run ITK's ScalarImageKmeansImageFilter (SimpleITK binding).

    The filter processes the full volume; because out-of-ROI voxels were
    forced to 0 during preprocessing, they cluster into a single background
    label which we subsequently mask out using the ROI.
    """

    def run(
        self,
        image: sitk.Image,
        roi_mask: sitk.Image,
        initial_means: list[float],
        strategy: str,
    ) -> KMeansResult:
        kmeans = sitk.ScalarImageKmeansImageFilter()
        kmeans.SetUseNonContiguousLabels(False)
        kmeans.SetClassWithInitialMean([float(m) for m in initial_means])
        output = kmeans.Execute(image)
        output = sitk.Cast(output, sitk.sitkUInt8)
        output.CopyInformation(image)

        final_means = list(kmeans.GetFinalMeans())
        label_arr = VolumeIO.to_numpy(output).astype(np.int16)

        roi_arr = VolumeIO.to_numpy(roi_mask).astype(bool)
        roi_label_arr = label_arr.copy()
        roi_label_arr[~roi_arr] = -1

        return KMeansResult(
            k=len(initial_means),
            strategy=strategy,
            initial_means=[float(x) for x in initial_means],
            final_means=[float(x) for x in final_means],
            label_image=output,
            label_array=label_arr,
            roi_label_array=roi_label_arr,
        )
