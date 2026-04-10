#!/usr/bin/env python
"""
Segmentación K-Means — ScalarImageKmeansImageFilter (ITK).

Clasifica los vóxeles de cada imagen MR en clusters basados en intensidad.
Se prueban 2, 3 y 4 clases para cada imagen (máximo 4).

Referencia:
    https://examples.itk.org/src/segmentation/classifiers/clusterpixelsingrayscaleimage/documentation
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

# Número de clases a probar (máximo 4)
CLASSES_TO_TRY = [2, 3, 4]

# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------


def compute_initial_centroids(path: str, n_classes: int) -> list[float]:
    """
    Calcula centroides iniciales distribuidos uniformemente entre el mínimo
    y máximo de intensidades válidas (ignorando vóxeles con valor 0).

    Parámetros
    ----------
    path      : str — ruta al archivo NIfTI
    n_classes : int — número de clases

    Retorna
    -------
    list[float] — centroides iniciales
    """
    image = itk.imread(path, itk.F)
    arr = itk.array_from_image(image)
    valid_pixels = arr[arr > 0].flatten()
    min_val = float(valid_pixels.min())
    max_val = float(valid_pixels.max())
    centroids = np.linspace(min_val, max_val, n_classes).tolist()
    return centroids


def apply_kmeans(path: str, centroids: list[float]) -> tuple[itk.Image, list[float]]:
    """
    Aplica ScalarImageKmeansImageFilter sobre la imagen.

    Parámetros
    ----------
    path      : str         — ruta al archivo NIfTI
    centroids : list[float] — centroides iniciales para cada clase

    Retorna
    -------
    output_image  : itk.Image    — imagen segmentada con etiquetas
    final_means   : list[float]  — centroides finales calculados
    """
    InputImageType = itk.Image[itk.F, 3]

    # Leer imagen como float
    image = itk.imread(path, itk.F)

    # Instanciar el filtro siguiendo la API oficial
    kmeans_filter = itk.ScalarImageKmeansImageFilter[InputImageType].New()
    kmeans_filter.SetInput(image)
    kmeans_filter.SetUseNonContiguousLabels(True)

    # Agregar cada clase con su centroide inicial
    for centroid in centroids:
        kmeans_filter.AddClassWithInitialMean(centroid)

    kmeans_filter.Update()

    # Obtener centroides finales
    final_means = list(kmeans_filter.GetFinalMeans())

    return kmeans_filter.GetOutput(), final_means


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = os.path.join(RESULTS_DIR, "kmeans")
    os.makedirs(output_dir, exist_ok=True)

    for key, path in IMAGES.items():
        for n_classes in CLASSES_TO_TRY:
            print(f"\n{'='*60}")
            print(f"  Imagen: {key}")
            print(f"  Número de clases: {n_classes}")
            print(f"{'='*60}")

            # Calcular centroides iniciales
            centroids = compute_initial_centroids(path, n_classes)
            centroids_str = ", ".join(f"{c:.2f}" for c in centroids)
            print(f"  Centroides iniciales: [{centroids_str}]")

            # Aplicar filtro K-Means
            result, final_means = apply_kmeans(path, centroids)

            # Imprimir centroides finales
            final_str = ", ".join(f"{m:.2f}" for m in final_means)
            print(f"  Centroides finales:   [{final_str}]")

            # Imprimir etiquetas generadas
            labels = list(range(n_classes))
            print(f"  Etiquetas generadas:  {labels}")

            # Guardar resultado
            out_path = os.path.join(output_dir, f"{key}_kmeans_{n_classes}.nii.gz")
            itk.imwrite(result, out_path)
            print(f"  Guardado: {out_path}")

    print("\n  Segmentación K-Means completada.")


if __name__ == "__main__":
    main()
