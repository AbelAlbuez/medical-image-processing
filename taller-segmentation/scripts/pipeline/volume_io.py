"""VolumeIO — reads/writes 3D float volumes while preserving ITK geometry.

Backed by SimpleITK (the Python-wrapped ITK toolkit) because the itk-python
C-extension DLLs are blocked by Windows Application Control on this host.
SimpleITK exposes the same ITK filters with the same numerical behaviour.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import SimpleITK as sitk


_STAGE_ROOT = Path(tempfile.gettempdir()) / "medseg_stage"
_STAGE_ROOT.mkdir(parents=True, exist_ok=True)


def _needs_staging(path: Path) -> bool:
    return os.name == "nt" and not str(path).isascii()


def _stage_for_read(path: Path) -> tuple[str, Path | None]:
    """Copy ``path`` to an ASCII temp file if necessary; return (read_path, tmp)."""
    if not _needs_staging(path):
        return str(path), None
    suffix = "".join(path.suffixes)
    tmp = _STAGE_ROOT / f"in_{uuid.uuid4().hex}{suffix}"
    shutil.copyfile(path, tmp)
    return str(tmp), tmp


def _stage_for_write(path: Path) -> tuple[str, Path | None, Path]:
    """Return ASCII temp write target + cleanup handle + final destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _needs_staging(path):
        return str(path), None, path
    suffix = "".join(path.suffixes)
    tmp = _STAGE_ROOT / f"out_{uuid.uuid4().hex}{suffix}"
    return str(tmp), tmp, path


class VolumeIO:
    """Load/save NIfTI/MHA volumes and move between SimpleITK and NumPy safely."""

    @staticmethod
    def read(path: str | Path) -> sitk.Image:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Volume not found: {path}")
        read_path, tmp = _stage_for_read(path)
        try:
            image = sitk.ReadImage(read_path, sitk.sitkFloat32)
        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        if image.GetDimension() != 3:
            raise ValueError(
                f"Expected 3D volume, got {image.GetDimension()}D: {path}"
            )
        return image

    @staticmethod
    def geometry(image: sitk.Image) -> dict:
        direction = list(image.GetDirection())
        direction_matrix = [direction[i * 3 : (i + 1) * 3] for i in range(3)]
        return {
            "origin": list(image.GetOrigin()),
            "spacing": list(image.GetSpacing()),
            "direction": direction_matrix,
            "size": list(image.GetSize()),
        }

    @staticmethod
    def copy_geometry(target: sitk.Image, reference: sitk.Image) -> sitk.Image:
        target.CopyInformation(reference)
        return target

    @staticmethod
    def to_numpy(image: sitk.Image) -> np.ndarray:
        return sitk.GetArrayFromImage(image).copy()

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        reference: sitk.Image,
        pixel_type: int = sitk.sitkFloat32,
    ) -> sitk.Image:
        if pixel_type == sitk.sitkFloat32:
            array = array.astype(np.float32, copy=False)
        elif pixel_type == sitk.sitkUInt8:
            array = array.astype(np.uint8, copy=False)
        elif pixel_type == sitk.sitkUInt16:
            array = array.astype(np.uint16, copy=False)
        elif pixel_type == sitk.sitkInt16:
            array = array.astype(np.int16, copy=False)
        image = sitk.GetImageFromArray(array)
        image.CopyInformation(reference)
        if image.GetPixelID() != pixel_type:
            image = sitk.Cast(image, pixel_type)
        return image

    @staticmethod
    def write(image: sitk.Image, path: str | Path) -> None:
        path = Path(path)
        write_path, tmp, final = _stage_for_write(path)
        sitk.WriteImage(image, write_path, useCompression=True)
        if tmp is not None:
            shutil.move(str(tmp), str(final))
