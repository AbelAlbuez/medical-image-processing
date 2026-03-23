#!/usr/bin/env python
"""
Comparacion cuantitativa de filtros — PSNR y MSE (Ali 2018, Tables 1 & 2).

Replica la metodologia del articulo:
  - Referencia: imagen con minimo ruido (pn1 o cualquier volumen "limpio")
  - Prueba:     se inyecta ruido sintetico sobre la referencia
  - Metrica:    PSNR = 10 * log10(MAX^2 / MSE)  [dB]

Uso:
    python src/metrics.py <reference.nii.gz> [opciones]

Ejemplos:
    # Tabla Gaussiano (replica Table 1 del articulo):
    python src/metrics.py t1_icbm_normal_1mm_pn1_rf20.nii.gz --mode gaussian

    # Tabla sal y pimienta (replica Table 2):
    python src/metrics.py t1_icbm_normal_1mm_pn1_rf20.nii.gz --mode salt_pepper

    # Ambas tablas + CSV de resultados:
    python src/metrics.py t1_icbm_normal_1mm_pn1_rf20.nii.gz --mode both --save-csv

Salida:
    Tabla de PSNR en pantalla.
    Si --save-csv: output/metrics/<base>_<mode>_metrics.csv
    Si --save-png: output/metrics/<base>_<mode>_metrics.png (grafico de barras)
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter, median_filter
import itk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "Images"
OUTPUT_METRICS_DIR = PROJECT_ROOT / "output" / "metrics"

DIMENSION = 3
UC_TYPE = itk.Image[itk.UC, DIMENSION]
F_TYPE  = itk.Image[itk.F,  DIMENSION]

_TMP_DIR: Optional[Path] = None


# -- Path / I/O helpers -------------------------------------------------------

def resolve_input_path(user_path: str) -> Path:
    p = Path(user_path)
    if p.is_file():
        return p.resolve()
    alt = IMAGES_DIR / user_path
    if alt.is_file():
        return alt.resolve()
    print(f"ERROR: no se encontro: {user_path}", file=sys.stderr)
    sys.exit(1)


def _get_stem(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def _itk_safe(p: Path) -> str:
    """Copy to ASCII-path temp dir if path has non-ASCII chars (ITK C++ limitation)."""
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_metrics_"))
        dst = _TMP_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        return str(dst)


def load_as_uc(filepath: Path) -> np.ndarray:
    """Load a NIfTI volume as uint8 numpy array (Z,Y,X)."""
    safe = _itk_safe(filepath)
    reader = itk.ImageFileReader[UC_TYPE].New()
    reader.SetFileName(safe)
    reader.Update()
    return itk.array_from_image(reader.GetOutput())


def _save_nii(arr: np.ndarray, ref_reader, out_path: Path) -> None:
    """Save a uint8 numpy array to NIfTI, copying spatial metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = itk.image_from_array(arr.astype(np.uint8))
    img.CopyInformation(ref_reader.GetOutput())
    safe_out = _itk_safe(out_path)
    writer = itk.ImageFileWriter[UC_TYPE].New()
    writer.SetFileName(safe_out)
    writer.SetInput(img)
    writer.Update()
    if safe_out != str(out_path):
        shutil.move(safe_out, out_path)


# -- Filter implementations ---------------------------------------------------

def apply_median(arr: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Median filter via scipy.ndimage.median_filter (3-D, isotropic window).
    Window side = 2*radius+1, matching the ITK MedianImageFilter convention.
    Used here instead of ITK to avoid platform-specific DLL dependencies in metrics.
    """
    size = 2 * radius + 1
    return median_filter(arr.astype(np.uint8), size=size, mode="reflect")


def apply_adaptive_median_numpy(arr: np.ndarray, smax: int = 7) -> np.ndarray:
    """NumPy implementation of adaptive median filter (Ali 2018, Sec 2.2.2)."""
    if smax % 2 == 0:
        smax -= 1
    if smax < 3:
        smax = 3
    out_vol = np.zeros_like(arr, dtype=np.uint8)
    for z in range(arr.shape[0]):
        out_vol[z] = _adaptive_median_slice(arr[z], smax)
    return out_vol


def _adaptive_median_slice(sl: np.ndarray, smax: int) -> np.ndarray:
    h, w = sl.shape
    pad = smax // 2 + 1
    padded = np.pad(sl.astype(np.uint8), pad, mode="reflect")
    out = np.zeros((h, w), dtype=np.uint8)
    for yi in range(h):
        for xi in range(w):
            py, px = yi + pad, xi + pad
            win = 3
            zxy = int(sl[yi, xi])
            while True:
                r = win // 2
                neigh = padded[py - r: py + r + 1, px - r: px + r + 1]
                zmin, zmax = int(neigh.min()), int(neigh.max())
                zmed = int(np.median(neigh))
                if (zmed - zmin) > 0 and (zmed - zmax) < 0:
                    out[yi, xi] = zxy if (zxy - zmin) > 0 and (zxy - zmax) < 0 else zmed
                    break
                win += 2
                if win > smax:
                    out[yi, xi] = zmed
                    break
    return out


def apply_wiener(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Adaptive Wiener filter (Ali 2018, Sec 2.1) using SciPy uniform_filter."""
    arr_f = arr.astype(np.float64)
    local_mean = uniform_filter(arr_f, size=window, mode="reflect")
    local_mean_sq = uniform_filter(arr_f ** 2, size=window, mode="reflect")
    local_var = local_mean_sq - local_mean ** 2
    nu2 = float(np.mean(local_var))
    eps = np.finfo(np.float64).eps
    gain = np.maximum(local_var - nu2, 0.0) / np.maximum(local_var, eps)
    return (local_mean + gain * (arr_f - local_mean)).astype(np.float32)


# -- Noise injection ----------------------------------------------------------

def inject_salt_pepper(arr: np.ndarray, density: float) -> np.ndarray:
    out = arr.copy().astype(np.uint8)
    n = int(out.size * density)
    if n <= 0:
        return out
    idx = np.random.choice(out.size, n, replace=False)
    out.flat[idx[: n // 2]] = 255
    out.flat[idx[n // 2 :]] = 0
    return out


def inject_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# -- PSNR metric --------------------------------------------------------------

def psnr(ref: np.ndarray, test: np.ndarray, max_val: float = 255.0) -> float:
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


def mse(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))


# -- Table helpers ------------------------------------------------------------

def _print_table(title: str, rows: list[dict]) -> None:
    """Print a PSNR/MSE comparison table to stdout."""
    if not rows:
        return
    col_keys = list(rows[0].keys())
    col_w = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in col_keys}
    sep = "+-" + "-+-".join("-" * col_w[k] for k in col_keys) + "-+"
    header = "| " + " | ".join(k.ljust(col_w[k]) for k in col_keys) + " |"
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(row[k]).ljust(col_w[k]) for k in col_keys) + " |")
    print(sep)


def _save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [csv] {out_path}")


def _save_bar_chart(rows: list[dict], title: str, out_path: Path) -> None:
    """Bar chart of PSNR values for each noise level × filter."""
    psnr_keys = [k for k in rows[0].keys() if "PSNR" in k and k != "PSNR_noisy"]
    levels = [r["noise_level"] for r in rows]
    x = np.arange(len(levels))
    width = 0.8 / len(psnr_keys)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(psnr_keys):
        vals = [float(r[key]) if r[key] != "inf" else 0 for r in rows]
        ax.bar(x + i * width, vals, width, label=key.replace("PSNR_", ""))

    # Also plot the noisy baseline
    noisy_vals = [float(r["PSNR_noisy"]) if r["PSNR_noisy"] != "inf" else 0 for r in rows]
    ax.plot(x + (len(psnr_keys) - 1) * width / 2, noisy_vals,
            "k--o", label="Noisy (no filter)", linewidth=1.5)

    ax.set_xlabel("Noise level")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.set_xticks(x + (len(psnr_keys) - 1) * width / 2)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [png] {out_path}")


# -- Main experiment runners --------------------------------------------------

def run_gaussian_table(ref_arr: np.ndarray,
                        sigmas: list[float], save_csv: bool,
                        save_png: bool, base: str) -> None:
    """Table 1 equivalent: Gaussian noise at different sigma levels."""
    rows = []
    print("\nRunning Gaussian noise experiments...")
    for sigma in sigmas:
        print(f"  sigma={sigma:.0f} ...", end=" ", flush=True)
        noisy = inject_gaussian(ref_arr, sigma)
        med = apply_median(noisy, radius=3)
        adm = apply_adaptive_median_numpy(noisy, smax=7)
        wie = apply_wiener(noisy, window=5)

        row = {
            "noise_level": f"sigma={sigma:.0f}",
            "PSNR_noisy":  f"{psnr(ref_arr, noisy):.2f}",
            "PSNR_median": f"{psnr(ref_arr, med):.2f}",
            "PSNR_adap_median": f"{psnr(ref_arr, adm):.2f}",
            "PSNR_wiener": f"{psnr(ref_arr, wie):.2f}",
            "MSE_noisy":   f"{mse(ref_arr, noisy):.2f}",
            "MSE_median":  f"{mse(ref_arr, med):.2f}",
            "MSE_adap_median": f"{mse(ref_arr, adm):.2f}",
            "MSE_wiener":  f"{mse(ref_arr, wie):.2f}",
        }
        rows.append(row)
        print("done")

    _print_table("=== Gaussian Noise — PSNR (dB) / MSE ===", rows)
    if save_csv:
        _save_csv(rows, OUTPUT_METRICS_DIR / f"{base}_gaussian_metrics.csv")
    if save_png:
        _save_bar_chart(rows, f"{base} - Gaussian Noise: Filter PSNR Comparison",
                        OUTPUT_METRICS_DIR / f"{base}_gaussian_psnr.png")


def run_saltpepper_table(ref_arr: np.ndarray,
                          densities: list[float], save_csv: bool,
                          save_png: bool, base: str) -> None:
    """Table 2 equivalent: Salt & Pepper noise at different densities."""
    rows = []
    print("\nRunning Salt & Pepper noise experiments...")
    for density in densities:
        print(f"  density={density:.2f} ...", end=" ", flush=True)
        noisy = inject_salt_pepper(ref_arr, density)
        med = apply_median(noisy, radius=3)
        adm = apply_adaptive_median_numpy(noisy, smax=7)
        wie = apply_wiener(noisy, window=5)

        row = {
            "noise_level": f"sp={density:.2f}",
            "PSNR_noisy":  f"{psnr(ref_arr, noisy):.2f}",
            "PSNR_median": f"{psnr(ref_arr, med):.2f}",
            "PSNR_adap_median": f"{psnr(ref_arr, adm):.2f}",
            "PSNR_wiener": f"{psnr(ref_arr, wie):.2f}",
            "MSE_noisy":   f"{mse(ref_arr, noisy):.2f}",
            "MSE_median":  f"{mse(ref_arr, med):.2f}",
            "MSE_adap_median": f"{mse(ref_arr, adm):.2f}",
            "MSE_wiener":  f"{mse(ref_arr, wie):.2f}",
        }
        rows.append(row)
        print("done")

    _print_table("=== Salt & Pepper Noise — PSNR (dB) / MSE ===", rows)
    if save_csv:
        _save_csv(rows, OUTPUT_METRICS_DIR / f"{base}_saltpepper_metrics.csv")
    if save_png:
        _save_bar_chart(rows, f"{base} - Salt & Pepper: Filter PSNR Comparison",
                        OUTPUT_METRICS_DIR / f"{base}_saltpepper_psnr.png")


# -- CLI ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparacion cuantitativa PSNR/MSE de los 3 filtros del taller."
    )
    parser.add_argument(
        "reference",
        help="Imagen de referencia ('limpia') — ruta o nombre dentro de Images/",
    )
    parser.add_argument(
        "--mode",
        choices=["gaussian", "salt_pepper", "both"],
        default="both",
        help="Tipo de experimento (por defecto: both)",
    )
    parser.add_argument(
        "--gaussian-sigmas",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0, 40.0],
        metavar="S",
        help="Niveles sigma Gaussiano (por defecto: 5 10 20 40)",
    )
    parser.add_argument(
        "--sp-densities",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.30],
        metavar="D",
        help="Densidades sal y pimienta (por defecto: 0.05 0.10 0.20 0.30)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Guardar resultados en CSV dentro de output/metrics/",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Guardar grafico de barras PNG dentro de output/metrics/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (por defecto: 42)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    ref_path = resolve_input_path(args.reference)
    base = _get_stem(ref_path)
    print(f"\nReferencia : {ref_path.name}")
    print(f"Modo       : {args.mode}")

    # Load reference
    safe = _itk_safe(ref_path)
    ref_reader = itk.ImageFileReader[UC_TYPE].New()
    ref_reader.SetFileName(safe)
    ref_reader.Update()
    ref_arr = itk.array_from_image(ref_reader.GetOutput())
    print(f"Shape      : {ref_arr.shape}  dtype={ref_arr.dtype}  range=[{ref_arr.min()},{ref_arr.max()}]")

    if args.mode in ("gaussian", "both"):
        run_gaussian_table(
            ref_arr,
            sigmas=args.gaussian_sigmas,
            save_csv=args.save_csv,
            save_png=args.save_png,
            base=base,
        )

    if args.mode in ("salt_pepper", "both"):
        run_saltpepper_table(
            ref_arr,
            densities=args.sp_densities,
            save_csv=args.save_csv,
            save_png=args.save_png,
            base=base,
        )

    print("\n  Metricas completas.")

    if _TMP_DIR is not None and _TMP_DIR.exists():
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
