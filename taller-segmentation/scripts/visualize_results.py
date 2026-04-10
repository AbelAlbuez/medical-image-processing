#!/usr/bin/env python
"""
Visualización de vistas ortogonales (axial, coronal, sagital) de volúmenes 3D.

Carga un volumen .nii.gz y genera una figura con las 3 vistas lado a lado.
Configurar IMAGE_PATH y los índices de slice al inicio del script.
"""
from __future__ import annotations

import os

import itk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuración — modificar según lo que se quiera visualizar
# ---------------------------------------------------------------------------

# Ruta al volumen a visualizar (puede ser imagen original o resultado de segmentación)
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "otsu", "brain_otsu_2.nii.gz")

# Índices de slice — si es None, se usa el slice central automáticamente
# NOTA: para cáncer de mama el tumor puede no estar en slices centrales;
#       ajustar SLICE_AXIAL manualmente hasta encontrar el corte con el tumor
SLICE_AXIAL   = None
SLICE_CORONAL = None
SLICE_SAGITAL = None

# Colormap: 'gray' para imágenes originales, 'tab10' para segmentaciones
COLORMAP = "tab10"

# ---------------------------------------------------------------------------
# Configuración de rutas de salida
# ---------------------------------------------------------------------------

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------


def load_volume(path: str) -> np.ndarray:
    """Carga un volumen NIfTI y lo convierte a numpy array."""
    image = itk.imread(path, itk.F)
    arr = itk.array_from_image(image)
    return arr


def get_slice_index(arr: np.ndarray, axis: int, requested: int | None) -> int:
    """Retorna el índice de slice: el solicitado o el central si es None."""
    if requested is not None:
        return requested
    return arr.shape[axis] // 2


def visualize_orthogonal(arr: np.ndarray, name: str, cmap: str) -> None:
    """
    Genera una figura con las 3 vistas ortogonales del volumen.

    Parámetros
    ----------
    arr  : np.ndarray — volumen 3D (Z, Y, X)
    name : str        — nombre para títulos y archivo de salida
    cmap : str        — colormap de matplotlib
    """
    # Calcular índices de slice
    idx_axial   = get_slice_index(arr, 0, SLICE_AXIAL)
    idx_coronal = get_slice_index(arr, 1, SLICE_CORONAL)
    idx_sagital = get_slice_index(arr, 2, SLICE_SAGITAL)

    # Extraer slices
    slice_axial   = arr[idx_axial, :, :]
    slice_coronal = arr[:, idx_coronal, :]
    slice_sagital = arr[:, :, idx_sagital]

    # Crear figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    titles = [
        f"Axial (slice {idx_axial})",
        f"Coronal (slice {idx_coronal})",
        f"Sagital (slice {idx_sagital})",
    ]
    slices = [slice_axial, slice_coronal, slice_sagital]

    for ax, title, sl in zip(axes, titles, slices):
        ax.imshow(sl, cmap=cmap, aspect="auto")
        ax.set_title(f"{name} — {title}", fontsize=12)
        ax.axis("off")

    fig.suptitle(f"Vistas ortogonales: {name}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    # Guardar figura
    out_path = os.path.join(FIGURES_DIR, f"{name}_vistas.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Verificar que el archivo existe
    if not os.path.isfile(IMAGE_PATH):
        print(f"  ERROR: no se encontró el archivo: {IMAGE_PATH}")
        print("  Asegúrese de haber ejecutado primero el script de segmentación correspondiente.")
        return

    print(f"  Cargando: {IMAGE_PATH}")
    arr = load_volume(IMAGE_PATH)
    print(f"  Forma del volumen: {arr.shape}")

    # Nombre base para títulos y archivo de salida
    basename = os.path.splitext(os.path.splitext(os.path.basename(IMAGE_PATH))[0])[0]

    visualize_orthogonal(arr, basename, COLORMAP)
    print("\n  Visualización completada.")


if __name__ == "__main__":
    main()
