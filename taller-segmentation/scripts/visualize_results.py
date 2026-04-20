#!/usr/bin/env python
"""Visualización de vistas ortogonales (axial, coronal, sagital) de volúmenes 3D.

Genera un mosaico por cada imagen (brain, breast, liver) con:
    * Imagen original (gris)
    * ROI anatómica (binaria)
    * Mapa de etiquetas K-means (tab10)
    * Máscara final de lesión (binaria)

Usa SimpleITK porque el wrapper itk-python está bloqueado por Windows App Control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import VolumeIO  # noqa: E402

ROOT = HERE.parent
IMAGES_DIR = ROOT / "images"
RESULTS_DIR = ROOT / "results" / "unsupervised_kmeans"
FIGURES_DIR = ROOT / "report" / "figures"

VOLUMES = [
    ("brain",  IMAGES_DIR / "MRBrainTumor.nii.gz"),
    ("breast", IMAGES_DIR / "MRBreastCancer.nii.gz"),
    ("liver",  IMAGES_DIR / "MRLiverTumor.nii.gz"),
]


def load_array(path: Path) -> np.ndarray:
    return VolumeIO.to_numpy(VolumeIO.read(path))


def find_focus_slice(
    lesion: np.ndarray, roi: np.ndarray
) -> tuple[int, int, int]:
    """Pick slices centred on the lesion; fall back to ROI centroid, then volume centre."""
    for candidate in (lesion.astype(bool), roi.astype(bool)):
        if candidate.any():
            z = int(np.argmax(candidate.sum(axis=(1, 2))))
            y = int(np.argmax(candidate.sum(axis=(0, 2))))
            x = int(np.argmax(candidate.sum(axis=(0, 1))))
            return z, y, x
    z, y, x = lesion.shape
    return z // 2, y // 2, x // 2


def render_row(ax_row, volume: np.ndarray, cmap: str, title: str,
               z: int, y: int, x: int) -> None:
    slices = [
        (volume[z, :, :], f"Axial z={z}"),
        (volume[:, y, :], f"Coronal y={y}"),
        (volume[:, :, x], f"Sagital x={x}"),
    ]
    for ax, (sl, slab) in zip(ax_row, slices):
        ax.imshow(sl, cmap=cmap, aspect="auto")
        ax.set_title(f"{title} · {slab}", fontsize=10)
        ax.axis("off")


def visualize_volume(name: str, original_path: Path) -> None:
    out_dir = RESULTS_DIR / name
    roi_path = out_dir / "roi_mask.nii.gz"
    label_path = out_dir / "label_map.nii.gz"
    lesion_path = out_dir / "lesion_mask.nii.gz"

    missing = [p for p in (original_path, roi_path, label_path, lesion_path) if not p.exists()]
    if missing:
        print(f"  [SKIP] {name}: falta {missing[0].name}")
        return

    print(f"  Renderizando mosaico: {name}")
    original = load_array(original_path)
    roi = load_array(roi_path)
    labels = load_array(label_path)
    lesion = load_array(lesion_path)

    z, y, x = find_focus_slice(lesion, roi)

    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    render_row(axes[0], original, "gray", "Original", z, y, x)
    render_row(axes[1], roi, "gray", "ROI", z, y, x)
    render_row(axes[2], labels, "tab10", "Label map", z, y, x)
    render_row(axes[3], lesion, "gray", "Lesion mask", z, y, x)

    fig.suptitle(f"Segmentación no supervisada K-means · {name}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"{name}_mosaico_unsupervised.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in VOLUMES:
        visualize_volume(name, path)
    print("\n  Mosaicos generados.")


if __name__ == "__main__":
    main()
