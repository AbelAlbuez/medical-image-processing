#!/usr/bin/env python
"""
Segmentación Otsu con múltiples umbrales — OtsuMultipleThresholdsImageFilter (ITK).

Busca automáticamente los umbrales óptimos para separar las regiones de la imagen.
Se prueban 1, 2 y 3 umbrales para cada imagen.

Referencia:
    https://examples.itk.org/src/filtering/thresholding/thresholdanimageusingotsu/documentation
"""
from __future__ import annotations

import os

import itk
import numpy as np

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------

IMAGES_DIR = "/Users/abelalbuez/Documents/Maestria/Tercer Semestre/Proc Img Medicas/medical-image-processing/Images"

IMAGES = {
    "brain":  os.path.join(IMAGES_DIR, "MRBrainTumor.nii.gz"),
    "breast": os.path.join(IMAGES_DIR, "MRBreastCancer.nii.gz"),
    "liver":  os.path.join(IMAGES_DIR, "MRLiverTumor.nii.gz"),
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Número de umbrales a probar
THRESHOLDS_TO_TRY = [1, 2, 3]

# Número de bins del histograma interno de Otsu
NUM_HISTOGRAM_BINS = 128

# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------


def apply_otsu(path: str, n_thresholds: int) -> tuple[itk.Image, list[float]]:
    """
    Aplica OtsuMultipleThresholdsImageFilter sobre la imagen.

    Parámetros
    ----------
    path          : str — ruta al archivo NIfTI
    n_thresholds  : int — número de umbrales a calcular

    Retorna
    -------
    output_image  : itk.Image    — imagen segmentada con etiquetas
    thresholds    : list[float]  — umbrales calculados por Otsu
    """
    InputImageType = itk.Image[itk.F, 3]
    OutputImageType = itk.Image[itk.UC, 3]

    # Leer imagen como float
    image = itk.imread(path, itk.F)

    # Instanciar el filtro siguiendo la API oficial
    otsu_filter = itk.OtsuMultipleThresholdsImageFilter[InputImageType, OutputImageType].New()
    otsu_filter.SetInput(image)
    otsu_filter.SetNumberOfThresholds(n_thresholds)
    otsu_filter.SetNumberOfHistogramBins(NUM_HISTOGRAM_BINS)
    otsu_filter.SetLabelOffset(0)
    otsu_filter.Update()

    # Obtener umbrales calculados
    thresholds = list(otsu_filter.GetThresholds())

    return otsu_filter.GetOutput(), thresholds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = os.path.join(RESULTS_DIR, "otsu")
    os.makedirs(output_dir, exist_ok=True)

    for key, path in IMAGES.items():
        for n_thresh in THRESHOLDS_TO_TRY:
            print(f"\n{'='*60}")
            print(f"  Imagen: {key}")
            print(f"  Número de umbrales: {n_thresh}")
            print(f"{'='*60}")

            # Aplicar filtro Otsu
            result, thresholds = apply_otsu(path, n_thresh)

            # Imprimir umbrales calculados
            thresh_str = ", ".join(f"{t:.2f}" for t in thresholds)
            print(f"  Umbrales calculados: [{thresh_str}]")

            # Guardar resultado
            out_path = os.path.join(output_dir, f"{key}_otsu_{n_thresh}.nii.gz")
            itk.imwrite(result, out_path)
            print(f"  Guardado: {out_path}")

    print("\n  Segmentación Otsu completada.")


if __name__ == "__main__":
    main()
