#!/usr/bin/env python
"""
Taller - seccion 2.1: Adaptive Wiener Filter (filtro de Wiener adaptativo).

Based on:
    Hanafy M. Ali, "MRI Medical Image Denoising by Fundamental Filters",
    Chapter 7 in High-Resolution Neuroimaging - Basic Physical Principles
    and Clinical Applications, IntechOpen, 2018.
    DOI: 10.5772/intechopen.72427

Algorithm (per-voxel):
    f_hat = mu + max(0, var_local - nu2) / max(var_local, eps) * (g - mu)

    mu        -- local mean over a cubic window of side `window`
    var_local -- local variance (E[X^2] - E[X]^2) over the same window
    nu2       -- global noise variance (mean of all local variances, or user-supplied)
    g         -- noisy input voxel
    eps       -- machine epsilon (avoids division by zero)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import shutil
import tempfile
import numpy as np
from scipy.ndimage import uniform_filter
import itk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project paths (same convention as median.py / adaptive-median.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "Images"
OUTPUT_WIENER_DIR = PROJECT_ROOT / "output" / "wiener"
OUTPUT_COMPARISON_DIR = PROJECT_ROOT / "output" / "comparison_results"

# ITK type for float32 volumes
DIMENSION = 3
PIXEL_TYPE = itk.F
IMAGE_TYPE = itk.Image[PIXEL_TYPE, DIMENSION]


# -- Path helpers -------------------------------------------------------------

def resolve_input_path(user_path: str) -> Path:
    """Resolve input: absolute path or filename inside Images/."""
    p = Path(user_path)
    if p.is_file():
        return p.resolve()
    alt = IMAGES_DIR / user_path
    if alt.is_file():
        return alt.resolve()
    print(f"ERROR: no se encontro la imagen de entrada: {user_path}", file=sys.stderr)
    print(f"  Buscado tambien en: {alt}", file=sys.stderr)
    sys.exit(1)


def _get_stem(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def default_output_path(input_path: Path) -> Path:
    OUTPUT_WIENER_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_WIENER_DIR / f"{_get_stem(input_path)}_wiener.nii"


# -- ITK I/O ------------------------------------------------------------------

_TMP_DIR: Optional[Path] = None  # cleaned up at end of main()


def _itk_safe_path(p: Path) -> str:
    """
    ITK's C++ layer cannot handle non-ASCII characters in file paths on Windows.
    When the path contains non-ASCII chars, copy the file to a temp directory
    that has an ASCII-only path while preserving the original filename (so ITK
    can still detect the format from the extension, e.g. '.nii.gz').
    Returns the safe path as a string.
    """
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s  # path is already ASCII-safe
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_wiener_"))
        dst = _TMP_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        return str(dst)


def _itk_safe_output(p: Path) -> str:
    """
    Return an ASCII-safe output path for ITK.  If the real output path is
    non-ASCII, ITK writes to a temp path; the caller must then move the file.
    """
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_wiener_"))
        return str(_TMP_DIR / p.name)


def load_as_float(filepath: str) -> itk.Image:
    """Read any 3-D medical image and cast pixels to float32 via CastImageFilter."""
    safe = _itk_safe_path(Path(filepath))
    reader = itk.ImageFileReader[itk.Image[itk.UC, DIMENSION]].New()
    reader.SetFileName(safe)
    reader.Update()
    raw = reader.GetOutput()
    cast = itk.CastImageFilter[itk.Image[itk.UC, DIMENSION], IMAGE_TYPE].New()
    cast.SetInput(raw)
    cast.Update()
    img = cast.GetOutput()
    size = list(img.GetLargestPossibleRegion().GetSize())
    spacing = [round(s, 4) for s in img.GetSpacing()]
    print(f"    Loaded  : {filepath}")
    print(f"    Size    : {size}  (X x Y x Z voxels)")
    print(f"    Spacing : {spacing}  mm/voxel")
    return img


def numpy_to_itk(arr: np.ndarray, reference: itk.Image) -> itk.Image:
    """Wrap float32 numpy array back to itk.Image, copying spatial metadata."""
    out = itk.image_from_array(arr.astype(np.float32))
    out.SetSpacing(reference.GetSpacing())
    out.SetOrigin(reference.GetOrigin())
    out.SetDirection(reference.GetDirection())
    return out


def save_image(itk_img: itk.Image, filepath: str) -> None:
    real = Path(filepath)
    safe = _itk_safe_output(real)
    writer = itk.ImageFileWriter[IMAGE_TYPE].New()
    writer.SetFileName(safe)
    writer.SetInput(itk_img)
    writer.Update()
    # If ITK wrote to a temp path, move the file to the real destination
    if safe != str(real):
        real.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(safe, real)
    print(f"    Guardado: {filepath}")


# -- Core algorithm -----------------------------------------------------------

def adaptive_wiener_filter_3d(
    arr: np.ndarray,
    window_size: int = 5,
    noise_variance: Optional[float] = None,
) -> np.ndarray:
    """
    Apply the 3-D Adaptive Wiener Filter (Ali 2018, Section 2.1).

    Parameters
    ----------
    arr : np.ndarray  shape (Z, Y, X), any float dtype
    window_size : int  -- side of the cubic neighbourhood window (odd)
    noise_variance : float or None  -- nu2; estimated from image if None

    Returns
    -------
    np.ndarray  shape (Z, Y, X), dtype float32
    """
    arr_f = arr.astype(np.float64)

    # Step 1 -- local mean mu  (box filter)
    local_mean = uniform_filter(arr_f, size=window_size, mode="reflect")

    # Step 2 -- local variance = E[X^2] - E[X]^2
    local_mean_sq = uniform_filter(arr_f ** 2, size=window_size, mode="reflect")
    local_var = local_mean_sq - local_mean ** 2

    # Step 3 -- global noise variance nu2
    if noise_variance is None:
        noise_variance = float(np.mean(local_var))
        print(f"    Auto-estimated nu2 = {noise_variance:.4f}")
    else:
        print(f"    Supplied       nu2 = {noise_variance:.4f}")

    # Step 4 -- Wiener gain in [0, 1] per voxel
    eps = np.finfo(np.float64).eps
    gain = np.maximum(local_var - noise_variance, 0.0) / np.maximum(local_var, eps)

    # Step 5 -- restore: f_hat = mu + gain * (g - mu)
    filtered = local_mean + gain * (arr_f - local_mean)
    return filtered.astype(np.float32)


# -- PSNR metric (from paper, Section 4) -------------------------------------

def compute_psnr(reference: np.ndarray, filtered: np.ndarray,
                 max_val: Optional[float] = None) -> float:
    if max_val is None:
        max_val = float(np.max(reference))
    mse = float(np.mean(
        (reference.astype(np.float64) - filtered.astype(np.float64)) ** 2
    ))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


# -- Noise injection (for experiments) ----------------------------------------

def inject_noise(arr: np.ndarray, noise_type: str,
                 density: float, sigma: float) -> np.ndarray:
    if noise_type == "salt_pepper":
        out = arr.copy()
        n = int(out.size * density)
        if n > 0:
            idx = np.random.choice(out.size, n, replace=False)
            out.flat[idx[: n // 2]] = float(arr.max())
            out.flat[idx[n // 2 :]] = float(arr.min())
        return out
    if noise_type == "gaussian":
        return arr + np.random.normal(0.0, sigma, arr.shape).astype(arr.dtype)
    if noise_type == "mixed":
        tmp = arr + np.random.normal(0.0, sigma, arr.shape).astype(arr.dtype)
        n = int(tmp.size * density)
        if n > 0:
            idx = np.random.choice(tmp.size, n, replace=False)
            tmp.flat[idx[: n // 2]] = float(arr.max())
            tmp.flat[idx[n // 2 :]] = float(arr.min())
        return tmp
    return arr.copy()


# -- PNG comparison output ----------------------------------------------------

def _three_views(arr: np.ndarray) -> tuple:
    """Axial, sagittal, coronal middle slices from a (Z,Y,X) volume."""
    z, y, x = arr.shape
    return arr[z // 2, :, :], arr[:, :, x // 2], arr[:, y // 2, :]


def save_comparison_png(
    orig_arr: np.ndarray,
    noisy_arr: np.ndarray,
    filtered_arr: np.ndarray,
    title: str,
    out_path: Path,
    noise_type: str = "none",
) -> None:
    """
    Guarda PNG con 3 vistas anatomicas (axial, sagittal, coronal):
      - Sin ruido: 2 columnas  -> Original | Wiener (AWF)
      - Con ruido: 3 columnas  -> Original | Ruidosa | Wiener (AWF)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if noise_type == "none":
        volumes = [orig_arr, filtered_arr]
        col_titles = ["Original", "Wiener (AWF)"]
    else:
        volumes = [orig_arr, noisy_arr, filtered_arr]
        col_titles = ["Original", f"Ruido ({noise_type})", "Wiener (AWF)"]

    n_cols = len(volumes)
    view_labels = ["Axial", "Sagittal", "Coronal"]
    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 3.5 * 3))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    for col, (arr, col_title) in enumerate(zip(volumes, col_titles)):
        views = _three_views(arr)
        for row, (view_label, sl) in enumerate(zip(view_labels, views)):
            ax = axes[row, col]
            ax.imshow(sl, cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_title, fontsize=9)
            if col == 0:
                ax.text(-0.05, 0.5, view_label, transform=ax.transAxes,
                        ha="right", va="center", fontsize=8, rotation=90)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [png] {out_path}")


# -- CLI ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive Wiener Filter 3D (Ali 2018, seccion 2.1). "
            "Entrada: volumen NIfTI; salida: NIfTI filtrado + PNG de comparacion."
        )
    )
    parser.add_argument(
        "input_image",
        help="Ruta al volumen de entrada o nombre de archivo dentro de Images/",
    )
    parser.add_argument(
        "output_image",
        nargs="?",
        default=None,
        help="Ruta de salida (.nii). Por defecto: output/wiener/<base>_wiener.nii",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        metavar="N",
        help="Lado de la ventana cubica NxNxN (impar; por defecto 5)",
    )
    parser.add_argument(
        "--noise-var",
        type=float,
        default=None,
        metavar="V",
        help="Varianza de ruido nu2 (por defecto: estimada automaticamente)",
    )
    parser.add_argument(
        "--noise-type",
        choices=["none", "salt_pepper", "gaussian", "mixed"],
        default="none",
        help="Tipo de ruido sintetico a inyectar antes de filtrar (por defecto 'none')",
    )
    parser.add_argument(
        "--noise-density",
        type=float,
        default=0.1,
        help="Fraccion de pixeles sal y pimienta (por defecto 0.1)",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=10.0,
        help="sigma del ruido Gaussiano (por defecto 10.0)",
    )
    parser.add_argument(
        "--reference",
        default=None,
        metavar="FILE",
        help="Imagen de referencia limpia para calcular PSNR (opcional)",
    )
    args = parser.parse_args()

    if args.window < 1:
        sys.exit("ERROR: --window debe ser >= 1")
    if args.window % 2 == 0:
        args.window += 1
        print(f"WARNING: --window ajustado a {args.window} (se requiere impar)")

    input_path = resolve_input_path(args.input_image)
    out_path = (
        Path(args.output_image) if args.output_image else default_output_path(input_path)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nAdaptive Wiener Filter - {input_path.name}")
    print(f"  Ventana : {args.window}x{args.window}x{args.window} voxels")
    print(f"  Salida  : {out_path}\n")

    # Load
    print("--- 1/5 Cargando imagen ---")
    itk_input = load_as_float(str(input_path))
    arr_in = itk.array_from_image(itk_input)

    # Noise injection (optional)
    print("--- 2/5 Inyectando ruido ---")
    noisy_arr = inject_noise(arr_in, args.noise_type, args.noise_density, args.noise_sigma)
    if args.noise_type == "none":
        print("    Sin ruido sintetico")
    else:
        print(f"    Ruido '{args.noise_type}' aplicado")

    # Filter
    print("--- 3/5 Aplicando Wiener adaptativo ---")
    arr_out = adaptive_wiener_filter_3d(
        noisy_arr, window_size=args.window, noise_variance=args.noise_var
    )
    print(f"    Rango de salida: [{arr_out.min():.2f}, {arr_out.max():.2f}]")

    # Save NIfTI
    print("--- 4/5 Guardando NIfTI ---")
    itk_output = numpy_to_itk(arr_out, reference=itk_input)
    save_image(itk_output, str(out_path))

    # Save PNG
    print("--- 5/5 Guardando PNG de comparacion ---")
    png_path = OUTPUT_COMPARISON_DIR / f"{_get_stem(input_path)}_wiener_comparison.png"
    save_comparison_png(
        arr_in,
        noisy_arr,
        arr_out,
        title=f"{_get_stem(input_path)} - noise={args.noise_type}  w={args.window}",
        out_path=png_path,
        noise_type=args.noise_type,
    )

    # PSNR (optional)
    if args.reference is not None:
        print("--- Bonus: PSNR ---")
        ref_itk = load_as_float(args.reference)
        ref_arr = itk.array_from_image(ref_itk)
        if ref_arr.shape == arr_out.shape:
            psnr_in = compute_psnr(ref_arr, noisy_arr)
            psnr_out = compute_psnr(ref_arr, arr_out)
            print(f"    PSNR entrada ruidosa : {psnr_in:.4f} dB")
            print(f"    PSNR filtrada (AWF)  : {psnr_out:.4f} dB")
            print(f"    Mejora               : {psnr_out - psnr_in:+.4f} dB")
        else:
            print(f"    WARNING: shape ref {ref_arr.shape} != output {arr_out.shape}; PSNR omitido")

    print("\n  Pipeline completo.")

    # Clean up temp directory used for ITK ASCII path workaround
    if _TMP_DIR is not None and _TMP_DIR.exists():
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
