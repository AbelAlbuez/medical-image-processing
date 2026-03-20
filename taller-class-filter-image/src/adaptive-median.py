#!/usr/bin/env python
"""
Taller — sección 2.2.2: Adaptive median filtering (base).

Implementar el método según el material del curso (p. ej. ventana adaptativa;
nota del enunciado: considerar tamaño 3×3 para el vecindario de mediana cuando aplique).

Este archivo solo define rutas, argparse y lectura/escritura ITK; el filtro adaptativo
va en el bloque TALLER.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import itk

# Raíz del proyecto taller-class-filter-image (padre de src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "Images"
OUTPUT_ADAPTIVE_MEDIAN_DIR = PROJECT_ROOT / "output" / "adaptive-median"


def resolve_input_path(user_path: str) -> Path:
    """
    Si la ruta no existe como está, intenta Images/<nombre>.
    Permite: `Images/foo.nii.gz`, `foo.nii.gz` o ruta absoluta.
    """
    p = Path(user_path)
    if p.is_file():
        return p.resolve()
    alt = IMAGES_DIR / user_path
    if alt.is_file():
        return alt.resolve()
    print(f"ERROR: no se encontró la imagen de entrada: {user_path}", file=sys.stderr)
    print(f"  Buscado también en: {alt}", file=sys.stderr)
    sys.exit(1)


def default_output_path(input_path: Path) -> Path:
    """Salida por defecto: output/adaptive-median/<base>_adaptive_median.nii"""
    name = input_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = input_path.stem
    OUTPUT_ADAPTIVE_MEDIAN_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ADAPTIVE_MEDIAN_DIR / f"{stem}_adaptive_median.nii"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive median filtering — 2.2.2 (taller, base)."
    )
    parser.add_argument(
        "input_image",
        help="Ruta al volumen de entrada o nombre de archivo dentro de Images/",
    )
    parser.add_argument(
        "output_image",
        nargs="?",
        default=None,
        help=(
            "Ruta de salida (.nii recomendado). "
            "Si se omite, se usa output/adaptive-median/<base>_adaptive_median.nii"
        ),
    )
    # Parámetros experimentales — ajustar según implementación (ej. tamaño máximo de ventana)
    parser.add_argument(
        "--max-window",
        type=int,
        default=None,
        help="(Opcional) parámetro reservado para la implementación del filtro adaptativo.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_image)
    if args.output_image is not None:
        out_path = Path(args.output_image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = default_output_path(input_path)

    PixelType = itk.UC
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]

    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(str(input_path))

    # --- TALLER: 2.2.2 Adaptive median filtering ---
    # Construir aquí el pipeline del filtro adaptativo (ITK manual, algoritmo propio, etc.).
    # Sustituir el passthrough inferior por la salida del filtro.
    # if args.max_window is not None:
    #     ...

    writer = itk.ImageFileWriter[ImageType].New()
    writer.SetFileName(str(out_path))
    # writer.SetInput(adaptiveMedianFilter.GetOutput())
    writer.SetInput(reader.GetOutput())  # temporal: sin filtro hasta implementar

    writer.Update()


if __name__ == "__main__":
    main()
