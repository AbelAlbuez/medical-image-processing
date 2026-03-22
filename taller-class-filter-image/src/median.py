#!/usr/bin/env python
"""
Filtro de mediana ITK (3D, unsigned char).
Si no se indica salida o la ruta no es absoluta, guarda en output/median/<base>_median_r{radius}.nii
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import itk

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
    """Busca la entrada en la ruta dada o en Images/."""
    p = Path(user_path)
    if p.is_file():
        return p.resolve()
    alt = PROJECT_ROOT / "Images" / user_path
    if alt.is_file():
        return alt.resolve()
    print(f"ERROR: no se encontró: {user_path}", file=sys.stderr)
    sys.exit(1)


def resolve_output_path(input_path: Path, output_arg: str | None, radius: int) -> Path:
    """
    Si no hay salida o la ruta no es absoluta → output/median/<base>_median_r{radius}.nii
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedianImageFilter (ITK, 3D). "
        "Uso: input radius [salida]; si se omite salida o no es absoluta → output/median/<base>_median_r{radius}.nii"
    )
    parser.add_argument("input_image")
    parser.add_argument("radius", type=int)
    parser.add_argument("output_image", nargs="?", default=None)
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_image)
    out_path = resolve_output_path(input_path, args.output_image, args.radius)

    PixelType = itk.UC
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]

    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(str(input_path))

    median_filter = itk.MedianImageFilter[ImageType, ImageType].New()
    median_filter.SetInput(reader.GetOutput())
    median_filter.SetRadius(args.radius)

    writer = itk.ImageFileWriter[ImageType].New()
    writer.SetFileName(str(out_path))
    writer.SetInput(median_filter.GetOutput())

    writer.Update()


if __name__ == "__main__":
    main()
