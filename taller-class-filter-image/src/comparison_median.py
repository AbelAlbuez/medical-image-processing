#!/usr/bin/env python
"""
Comparación visual: Original vs Mediana ITK vs Mediana adaptativa.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import itk

_TMP_DIR = None
_UC3 = itk.Image[itk.UC, 3]


def _itk_safe(p: Path) -> str:
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_cmp_"))
        dst = _TMP_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        return str(dst)


def _load_uc3(path: Path):
    r = itk.ImageFileReader[_UC3].New()
    r.SetFileName(_itk_safe(path))
    r.Update()
    return r.GetOutput()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
IMAGES_DIR = PROJECT_ROOT / "Images"
OUTPUT_MEDIAN = PROJECT_ROOT / "output" / "median"
OUTPUT_ADAPTIVE = PROJECT_ROOT / "output" / "adaptive-median"
OUTPUT_COMPARISON = PROJECT_ROOT / "output" / "comparison_results"


def resolve_input(user_path: str) -> Path:
    p = Path(user_path)
    if p.is_file():
        return p.resolve()
    alt = IMAGES_DIR / user_path
    if alt.is_file():
        return alt.resolve()
    print(f"ERROR: no se encontró la imagen: {user_path}", file=sys.stderr)
    sys.exit(1)


def basename_stem(p: Path) -> str:
    n = p.name
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return p.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara Original, mediana ITK y mediana adaptativa (PNG)."
    )
    parser.add_argument("input_image")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--max-window", type=int, default=7)
    parser.add_argument(
        "--no-itk",
        action="store_true",
        help="Reenvía --no-itk a adaptive-median.py (implementación NumPy).",
    )
    args = parser.parse_args()

    inp = resolve_input(args.input_image)
    base = basename_stem(inp)
    radius = args.radius
    smax = args.max_window

    OUTPUT_MEDIAN.mkdir(parents=True, exist_ok=True)
    OUTPUT_ADAPTIVE.mkdir(parents=True, exist_ok=True)
    OUTPUT_COMPARISON.mkdir(parents=True, exist_ok=True)

    out_median = OUTPUT_MEDIAN / f"{base}_median_r{radius}.nii"
    out_adaptive = OUTPUT_ADAPTIVE / f"{base}_adaptive_median.nii"
    out_png = OUTPUT_COMPARISON / f"{base}_comparison.png"

    py = sys.executable
    median_script = SRC_DIR / "median.py"
    adaptive_script = SRC_DIR / "adaptive-median.py"

    cmd_median = [
        py,
        str(median_script),
        str(inp),
        str(radius),
        str(out_median),
    ]
    cmd_adaptive = [
        py,
        str(adaptive_script),
        str(inp),
        str(out_adaptive),
        "--max-window",
        str(smax),
    ]
    if args.no_itk:
        cmd_adaptive.append("--no-itk")

    r1 = subprocess.run(cmd_median, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if r1.returncode != 0:
        print(r1.stderr or r1.stdout, file=sys.stderr)
        sys.exit(r1.returncode)

    r2 = subprocess.run(cmd_adaptive, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if r2.returncode != 0:
        print(r2.stderr or r2.stdout, file=sys.stderr)
        sys.exit(r2.returncode)

    vol_orig = _load_uc3(inp)
    vol_med = _load_uc3(out_median)
    vol_ad = _load_uc3(out_adaptive)

    def three_views(img):
        arr = itk.array_from_image(img)
        if arr.ndim != 3:
            raise ValueError("Se espera volumen 3D")
        z, y, x = arr.shape
        return arr[z // 2, :, :], arr[:, :, x // 2], arr[:, y // 2, :]

    col_titles = [
        "Original",
        f"Median Filter (r={radius})",
        "Adaptive Median (numpy)" if args.no_itk else "Adaptive Median (ITK)",
    ]
    volumes_views = [three_views(v) for v in (vol_orig, vol_med, vol_ad)]
    view_labels = ["Axial", "Sagittal", "Coronal"]

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for col, (col_title, views) in enumerate(zip(col_titles, volumes_views)):
        for row, (view_label, sl) in enumerate(zip(view_labels, views)):
            ax = axes[row, col]
            ax.imshow(sl, cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_title, fontsize=10)
            if col == 0:
                ax.text(-0.05, 0.5, view_label, transform=ax.transAxes,
                        ha="right", va="center", fontsize=9, rotation=90)
    fig.suptitle(f"{base} - Median vs Adaptive Median", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(out_png)


if __name__ == "__main__":
    try:
        main()
    finally:
        if _TMP_DIR is not None and Path(_TMP_DIR).exists():
            shutil.rmtree(_TMP_DIR, ignore_errors=True)
