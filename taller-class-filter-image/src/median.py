#!/usr/bin/env python
"""
Filtro de mediana 3D - implementacion propia (numpy).

Algoritmo (Ali 2018, seccion 2.2.1):
    Para cada voxel, tomar la vecindad (2*radius+1)^3, excluir el pixel
    central y reemplazar el voxel con la mediana de los vecinos restantes.

    neighborhood = volume[z-r:z+r+1, y-r:y+r+1, x-r:x+r+1]
    neighbors    = neighborhood.flatten()
    neighbors_without_center = np.delete(neighbors, center_index)
    output[z, y, x] = np.median(neighbors_without_center)

Si no se indica salida o la ruta no es absoluta, guarda en:
    output/median/<base>_median_r{radius}.nii
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import itk
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_TMP_DIR = None


def _itk_safe(p: Path) -> str:
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_med_"))
        dst = _TMP_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        return str(dst)


def _itk_safe_write(p: Path) -> str:
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_med_"))
        return str(_TMP_DIR / p.name)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_MEDIAN_DIR = PROJECT_ROOT / "output" / "median"


def _basename_from_path(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def resolve_input_path(user_path: str) -> Path:
    p = Path(user_path)
    if p.is_file():
        return p.resolve()
    alt = PROJECT_ROOT / "Images" / user_path
    if alt.is_file():
        return alt.resolve()
    print(f"ERROR: no se encontro: {user_path}", file=sys.stderr)
    sys.exit(1)


def resolve_output_path(input_path: Path, output_arg: str | None, radius: int) -> Path:
    base = _basename_from_path(input_path)
    if output_arg is None:
        OUTPUT_MEDIAN_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_MEDIAN_DIR / f"{base}_median_r{radius}.nii"
    outp = Path(output_arg)
    if outp.is_absolute():
        outp.parent.mkdir(parents=True, exist_ok=True)
        return outp
    OUTPUT_MEDIAN_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_MEDIAN_DIR / f"{base}_median_r{radius}.nii"


def median_filter_excl_center(arr: np.ndarray, radius: int) -> np.ndarray:
    """
    Median filter that excludes the center voxel from each neighborhood.

    Equivalent to the loop:
        for z, y, x in every voxel:
            neighborhood = arr[z-r:z+r+1, y-r:y+r+1, x-r:x+r+1].flatten()
            neighbors_without_center = np.delete(neighborhood, center_index)
            output[z, y, x] = np.median(neighbors_without_center)

    Implemented with sliding_window_view for speed (no Python loop per voxel).
    """
    w = 2 * radius + 1
    center = w ** 3 // 2  # index of center in flattened window (e.g. 13 for r=1)

    arr_f = arr.astype(np.float32)
    padded = np.pad(arr_f, radius, mode="reflect")

    # windows shape: (Z, Y, X, w, w, w)
    windows = sliding_window_view(padded, window_shape=(w, w, w))

    # flatten each window: (Z, Y, X, w^3)
    flat = windows.reshape(*arr.shape, w ** 3)

    # remove center pixel: (Z, Y, X, w^3 - 1)
    flat_no_center = np.delete(flat, center, axis=-1)

    # median along last axis
    return np.median(flat_no_center, axis=-1).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filtro de mediana 3D propio (numpy, excluye pixel central). "
        "Uso: input radius [salida]"
    )
    parser.add_argument("input_image")
    parser.add_argument("radius", type=int)
    parser.add_argument("output_image", nargs="?", default=None)
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_image)
    out_path = resolve_output_path(input_path, args.output_image, args.radius)

    ImageType = itk.Image[itk.UC, 3]

    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(_itk_safe(input_path))
    reader.Update()

    arr = itk.array_from_image(reader.GetOutput())
    print(f"  Loaded : {input_path.name}  shape={arr.shape}")

    filtered = median_filter_excl_center(arr, args.radius)
    print(f"  Filtered: radius={args.radius}, window={(2*args.radius+1)}^3, center excluded")

    out_img = itk.image_from_array(filtered)
    out_img.CopyInformation(reader.GetOutput())

    safe_out = _itk_safe_write(out_path)
    writer = itk.ImageFileWriter[ImageType].New()
    writer.SetFileName(safe_out)
    writer.SetInput(out_img)
    writer.Update()

    if safe_out != str(out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(safe_out, out_path)

    print(f"  Saved  : {out_path.name}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _TMP_DIR is not None and Path(_TMP_DIR).exists():
            shutil.rmtree(_TMP_DIR, ignore_errors=True)
