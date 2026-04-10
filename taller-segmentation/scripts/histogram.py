#!/usr/bin/env python
"""
Análisis de histogramas de intensidad para las imágenes MR.

Genera histogramas individuales y un comparativo lado a lado.
Imprime estadísticas descriptivas (min, max, media, std, percentiles).
"""
from __future__ import annotations

import os

import itk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------


def compute_histogram_stats(key: str, path: str) -> tuple[np.ndarray, dict]:
    """
    Carga la imagen, extrae vóxeles válidos (> 0) y calcula estadísticas.

    Retorna
    -------
    valid_pixels : np.ndarray — vóxeles aplanados sin fondo
    stats        : dict       — estadísticas descriptivas
    """
    print(f"\n{'='*60}")
    print(f"  Imagen: {key}  ({os.path.basename(path)})")
    print(f"{'='*60}")

    # Cargar imagen como float
    image = itk.imread(path, itk.F)
    arr = itk.array_from_image(image)
    print(f"  Forma del volumen: {arr.shape}")

    # Aplanar e ignorar vóxeles de fondo (valor 0)
    flat = arr.flatten()
    valid_pixels = flat[flat > 0]
    print(f"  Vóxeles totales: {len(flat):,}")
    print(f"  Vóxeles válidos (> 0): {len(valid_pixels):,}")

    # Calcular estadísticas
    percentiles = [25, 50, 75, 90, 95, 99]
    pvals = np.percentile(valid_pixels, percentiles)

    stats = {
        "min": float(valid_pixels.min()),
        "max": float(valid_pixels.max()),
        "mean": float(valid_pixels.mean()),
        "std": float(valid_pixels.std()),
    }
    for p, v in zip(percentiles, pvals):
        stats[f"p{p}"] = float(v)

    # Imprimir estadísticas
    print(f"  Mínimo:  {stats['min']:.2f}")
    print(f"  Máximo:  {stats['max']:.2f}")
    print(f"  Media:   {stats['mean']:.2f}")
    print(f"  Std:     {stats['std']:.2f}")
    for p in percentiles:
        print(f"  Percentil {p:>2}: {stats[f'p{p}']:.2f}")

    return valid_pixels, stats


def plot_single_histogram(key: str, valid_pixels: np.ndarray, stats: dict) -> None:
    """Genera y guarda el histograma individual de una imagen."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid_pixels, bins=100, color="steelblue", edgecolor="black", alpha=0.7)

    # Líneas de referencia en percentiles
    colors = {"p25": "green", "p50": "orange", "p75": "red", "p95": "purple"}
    for pkey, color in colors.items():
        ax.axvline(stats[pkey], color=color, linestyle="--", linewidth=1.5,
                    label=f"{pkey}: {stats[pkey]:.1f}")

    ax.set_title(f"Histograma de intensidades — {key}", fontsize=14)
    ax.set_xlabel("Intensidad", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(FIGURES_DIR, f"histogram_{key}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


def plot_comparative(data: dict) -> None:
    """Genera una figura comparativa con los 3 histogramas lado a lado."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key, (valid_pixels, stats)) in zip(axes, data.items()):
        ax.hist(valid_pixels, bins=100, color="steelblue", edgecolor="black", alpha=0.7)

        # Líneas de referencia
        colors = {"p25": "green", "p50": "orange", "p75": "red", "p95": "purple"}
        for pkey, color in colors.items():
            ax.axvline(stats[pkey], color=color, linestyle="--", linewidth=1.5,
                        label=f"{pkey}: {stats[pkey]:.1f}")

        ax.set_title(f"{key}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Intensidad", fontsize=11)
        ax.set_ylabel("Frecuencia", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Comparación de histogramas de intensidad", fontsize=15, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "histogram_comparativo.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figura comparativa guardada: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    data = {}
    for key, path in IMAGES.items():
        valid_pixels, stats = compute_histogram_stats(key, path)
        plot_single_histogram(key, valid_pixels, stats)
        data[key] = (valid_pixels, stats)

    plot_comparative(data)
    print("\n  Histogramas generados correctamente.")


if __name__ == "__main__":
    main()
