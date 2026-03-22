#!/usr/bin/env python
"""
Comparación visual: Original vs Mediana ITK vs Mediana adaptativa.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import itk
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

    vol_orig = itk.imread(str(inp))
    vol_med = itk.imread(str(out_median))
    vol_ad = itk.imread(str(out_adaptive))

    def middle_slice(img):
        arr = itk.array_from_image(img)
        if arr.ndim != 3:
            raise ValueError("Se espera volumen 3D")
        return arr[arr.shape[0] // 2, :, :]

    s0 = middle_slice(vol_orig)
    s1 = middle_slice(vol_med)
    s2 = middle_slice(vol_ad)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    titles = [
        "Original",
        f"Median Filter (r={radius})",
        "Adaptive Median (numpy)" if args.no_itk else "Adaptive Median (ITK)",
    ]
    for ax, sl, title in zip(axes, (s0, s1, s2), titles):
        ax.imshow(sl, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{base} — Median vs Adaptive Median", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(out_png)


if __name__ == "__main__":
    main()
