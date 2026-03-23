#!/usr/bin/env python
"""
run_all.py — Ejecutor por lotes del Taller 1.

Aplica los tres filtros (Mediana, Mediana Adaptativa, Wiener Adaptativo) sobre
cada imagen BrainWeb y genera los PNG de comparación para el informe.

Salidas generadas
-----------------
output/report/<base>_comparison.png   -- 4 columnas × 3 filas (axial/sagital/coronal)
output/report/summary_axial.png       -- todos los niveles de ruido × filtros (solo axial)

Uso
---
    python src/run_all.py
    python src/run_all.py --median-radius 1 --max-window 7 --wiener-window 5
    python src/run_all.py --force   # re-ejecuta aunque las salidas ya existan
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import itk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
IMAGES_DIR   = PROJECT_ROOT / "Images"
OUT_MEDIAN   = PROJECT_ROOT / "output" / "median"
OUT_ADAPTIVE = PROJECT_ROOT / "output" / "adaptive-median"
OUT_WIENER   = PROJECT_ROOT / "output" / "wiener"
OUT_REPORT   = PROJECT_ROOT / "output" / "report"

# Imágenes BrainWeb (T1, 1 mm isótropo, inhomogeneidad RF 20%)
IMAGES = [
    "t1_icbm_normal_1mm_pn3_rf20.nii.gz",
    "t1_icbm_normal_1mm_pn5_rf20.nii.gz",
    "t1_icbm_normal_1mm_pn9_rf20.nii.gz",
]

# Etiquetas de cada nivel de ruido para títulos y ejes
NOISE_LABELS = ["pn3 (3%)", "pn5 (5%)", "pn9 (9%)"]

# ---------------------------------------------------------------------------
# Manejo de rutas no-ASCII para ITK
# ---------------------------------------------------------------------------
_TMP_DIR: Path | None = None


def _itk_safe(p: Path) -> str:
    """
    Copia el archivo a un directorio temporal con ruta ASCII pura si la ruta
    contiene caracteres no-ASCII. La capa C++ de ITK en Windows no puede
    abrir rutas con tildes, ñ u otros caracteres especiales.
    """
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_run_"))
        dst = _TMP_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        return str(dst)


def _load(path: Path) -> np.ndarray:
    """Carga un volumen NIfTI como array NumPy uint8 con forma (Z, Y, X)."""
    ImageType = itk.Image[itk.UC, 3]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(_itk_safe(path))
    reader.Update()
    return itk.array_from_image(reader.GetOutput())


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def stem(p: Path) -> str:
    """Extrae el nombre base del archivo sin la extensión NIfTI (.nii o .nii.gz)."""
    n = p.name
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return p.stem


def three_views(arr: np.ndarray):
    """Devuelve los cortes centrales axial, sagital y coronal de un volumen (Z, Y, X)."""
    z, y, x = arr.shape
    return arr[z // 2, :, :], arr[:, :, x // 2], arr[:, y // 2, :]


def run_script(script: str, args: list[str], label: str) -> int:
    """
    Ejecuta un script Python como subproceso con los argumentos dados.
    Imprime el error en stderr si el proceso falla. Devuelve el código de retorno.
    """
    cmd = [sys.executable, str(SRC_DIR / script)] + args
    print(f"    ejecutando {label} ...", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    ERROR en {label}:\n{r.stderr or r.stdout}", file=sys.stderr)
    return r.returncode


# ---------------------------------------------------------------------------
# PNG de comparación por imagen (4 columnas × 3 filas)
# ---------------------------------------------------------------------------
def make_comparison_png(
    base: str,
    arrs: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """
    Genera el PNG de comparación completa para una imagen.
    Columnas: Original | Mediana | Mediana Adaptativa | Wiener Adaptativo.
    Filas: corte axial | sagital | coronal.
    """
    col_titles = list(arrs.keys())
    view_labels = ["Axial", "Sagital", "Coronal"]
    n_cols = len(col_titles)

    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 3.5 * 3))
    for col, title in enumerate(col_titles):
        views = three_views(arrs[title])
        for row, (view_label, sl) in enumerate(zip(view_labels, views)):
            ax = axes[row, col]
            ax.imshow(sl, cmap="gray", interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            if col == 0:
                ax.text(
                    -0.08, 0.5, view_label,
                    transform=ax.transAxes,
                    ha="right", va="center",
                    fontsize=9, rotation=90,
                )

    fig.suptitle(base, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    [PNG] {out_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Mosaico resumen (niveles de ruido como filas, filtros como columnas, solo axial)
# ---------------------------------------------------------------------------
def make_summary_png(
    all_data: list[dict[str, np.ndarray]],
    noise_labels: list[str],
    out_path: Path,
) -> None:
    """
    Genera el PNG resumen con todos los niveles de ruido × todos los filtros.
    Solo muestra el corte axial central. Cada fila es un nivel de ruido,
    cada columna es un filtro (o el original).
    """
    col_titles = list(all_data[0].keys())
    n_rows = len(all_data)
    n_cols = len(col_titles)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows)
    )
    # Garantizar array 2D de ejes incluso con una sola fila
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (data, noise_lbl) in enumerate(zip(all_data, noise_labels)):
        for col, title in enumerate(col_titles):
            ax = axes[row, col]
            axial = three_views(data[title])[0]  # solo el corte axial
            ax.imshow(axial, cmap="gray", interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
            if col == 0:
                # Etiqueta del nivel de ruido en el eje Y de la primera columna
                ax.set_ylabel(noise_lbl, fontsize=9)
                ax.yaxis.set_visible(True)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_yticks([])

    fig.suptitle("Todos los filtros × todos los niveles de ruido (corte axial)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[RESUMEN] {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aplica los tres filtros sobre todas las imágenes BrainWeb y genera los PNG del informe."
    )
    parser.add_argument("--median-radius", type=int, default=1,
                        help="Radio del filtro de mediana (por defecto 1)")
    parser.add_argument("--max-window",    type=int, default=7,
                        help="Ventana máxima del filtro de mediana adaptativa (por defecto 7)")
    parser.add_argument("--wiener-window", type=int, default=5,
                        help="Tamaño de ventana del filtro de Wiener (por defecto 5)")
    parser.add_argument("--force", action="store_true",
                        help="Re-ejecuta los filtros aunque las salidas ya existan")
    args = parser.parse_args()

    r  = args.median_radius
    mw = args.max_window
    ww = args.wiener_window

    # Crear directorios de salida
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    OUT_MEDIAN.mkdir(parents=True, exist_ok=True)
    OUT_ADAPTIVE.mkdir(parents=True, exist_ok=True)
    OUT_WIENER.mkdir(parents=True, exist_ok=True)

    all_data: list[dict[str, np.ndarray]] = []

    for img_name, noise_lbl in zip(IMAGES, NOISE_LABELS):
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            print(f"[OMITIDO] {img_name} no encontrado en Images/", flush=True)
            continue

        b = stem(img_path)
        print(f"\n=== {noise_lbl}  ({img_name}) ===", flush=True)

        # --- Filtro de Mediana ---
        med_out = OUT_MEDIAN / f"{b}_median_r{r}.nii"
        if args.force or not med_out.exists():
            rc = run_script("median.py",
                            [str(img_path), str(r), str(med_out)],
                            f"mediana radio={r}")
            if rc != 0:
                sys.exit(rc)
        else:
            print(f"    [omitido] mediana ya existe", flush=True)

        # --- Filtro de Mediana Adaptativa (algoritmo NumPy exacto de Ali 2018) ---
        adap_out = OUT_ADAPTIVE / f"{b}_adaptive_median.nii"
        if args.force or not adap_out.exists():
            rc = run_script("adaptive-median.py",
                            [str(img_path), str(adap_out),
                             "--max-window", str(mw), "--no-itk"],
                            f"mediana adaptativa mw={mw} (NumPy)")
            if rc != 0:
                sys.exit(rc)
        else:
            print(f"    [omitido] mediana adaptativa ya existe", flush=True)

        # --- Filtro de Wiener Adaptativo ---
        wien_out = OUT_WIENER / f"{b}_wiener.nii"
        if args.force or not wien_out.exists():
            rc = run_script("wiener.py",
                            [str(img_path), str(wien_out),
                             "--window", str(ww)],
                            f"wiener ventana={ww}")
            if rc != 0:
                sys.exit(rc)
        else:
            print(f"    [omitido] wiener ya existe", flush=True)

        # --- Cargar los cuatro volúmenes (original + tres filtrados) ---
        print(f"    cargando volúmenes ...", flush=True)
        arrs = {
            "Original":          _load(img_path),
            f"Mediana (r={r})":  _load(med_out),
            "Med. Adaptativa":   _load(adap_out),
            "Wiener Adaptativo": _load(wien_out),
        }

        # --- PNG de comparación completa por imagen ---
        png_out = OUT_REPORT / f"{b}_comparison.png"
        make_comparison_png(f"{noise_lbl} - {b}", arrs, png_out)

        all_data.append(arrs)

    # --- PNG mosaico resumen con todos los niveles de ruido ---
    if all_data:
        present_labels = [
            lbl for lbl, img in zip(NOISE_LABELS, IMAGES)
            if (IMAGES_DIR / img).exists()
        ]
        summary_out = OUT_REPORT / "summary_axial.png"
        make_summary_png(all_data, present_labels, summary_out)

    print("\nListo.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Limpiar directorio temporal para rutas no-ASCII
        if _TMP_DIR is not None and _TMP_DIR.exists():
            shutil.rmtree(_TMP_DIR, ignore_errors=True)
