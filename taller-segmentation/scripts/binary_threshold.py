#!/usr/bin/env python
"""
Segmentación por umbral binario — BinaryThresholdImageFilter (ITK).

Aplica un rango [lower, upper] a cada imagen MR para segmentar regiones
de interés. Los parámetros deben ajustarse según el análisis de histogramas.

Referencia:
    https://examples.itk.org/src/filtering/thresholding/thresholdanimageusingbinary/documentation
"""
from __future__ import annotations

import os

import itk
import numpy as np

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")

IMAGES = {
    "brain":  os.path.join(IMAGES_DIR, "MRBrainTumor.nii.gz"),
    "breast": os.path.join(IMAGES_DIR, "MRBreastCancer.nii.gz"),
    "liver":  os.path.join(IMAGES_DIR, "MRLiverTumor.nii.gz"),
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ---------------------------------------------------------------------------
# Parámetros de umbral por imagen (ajustar tras analizar histogramas)
# ---------------------------------------------------------------------------

PARAMS = {
    "brain":  {"lower": 100.0, "upper": 600.0},
    "breast": {"lower": 50.0,  "upper": 400.0},
    "liver":  {"lower": 80.0,  "upper": 500.0},
}

# Valores de la máscara binaria
INSIDE_VALUE = 1
OUTSIDE_VALUE = 0

# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------


def apply_binary_threshold(path: str, lower: float, upper: float) -> itk.Image:
    """
    Aplica BinaryThresholdImageFilter sobre la imagen en la ruta indicada.

    Parámetros
    ----------
    path  : str   — ruta al archivo NIfTI
    lower : float — umbral inferior del rango
    upper : float — umbral superior del rango

    Retorna
    -------
    itk.Image — imagen binaria segmentada
    """
    # Leer imagen como float
    InputImageType = itk.Image[itk.F, 3]
    OutputImageType = itk.Image[itk.F, 3]

    image = itk.imread(path, itk.F)

    # Instanciar el filtro siguiendo la API oficial
    threshold_filter = itk.BinaryThresholdImageFilter[InputImageType, OutputImageType].New()
    threshold_filter.SetInput(image)
    threshold_filter.SetLowerThreshold(lower)
    threshold_filter.SetUpperThreshold(upper)
    threshold_filter.SetInsideValue(INSIDE_VALUE)
    threshold_filter.SetOutsideValue(OUTSIDE_VALUE)
    threshold_filter.Update()

    return threshold_filter.GetOutput()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = os.path.join(RESULTS_DIR, "binary_threshold")
    os.makedirs(output_dir, exist_ok=True)

    for key, path in IMAGES.items():
        params = PARAMS[key]
        lower = params["lower"]
        upper = params["upper"]

        print(f"\n{'='*60}")
        print(f"  Imagen: {key}")
        print(f"  Umbral inferior: {lower:.1f}")
        print(f"  Umbral superior: {upper:.1f}")
        print(f"  Inside value: {INSIDE_VALUE}, Outside value: {OUTSIDE_VALUE}")
        print(f"{'='*60}")

        # Aplicar filtro
        result = apply_binary_threshold(path, lower, upper)

        # Guardar resultado
        out_path = os.path.join(output_dir, f"{key}_binary.nii.gz")
        itk.imwrite(result, out_path)
        print(f"  Guardado: {out_path}")

    print("\n  Segmentación por umbral binario completada.")


if __name__ == "__main__":
    main()
