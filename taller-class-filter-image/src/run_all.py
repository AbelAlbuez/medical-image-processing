#!/usr/bin/env python
"""
run_all.py
----------
Runs all 3 filters (Median, Adaptive Median, Adaptive Wiener) on every
BrainWeb noise-level image and produces comparison PNGs for the report.

Outputs
-------
output/report/<base>_comparison.png   -- 4 cols x 3 rows (axial/sag/coronal)
output/report/summary_axial.png       -- all noise levels x all filters (axial only)

Usage
-----
    python src/run_all.py
    python src/run_all.py --median-radius 1 --max-window 7 --wiener-window 5
    python src/run_all.py --force   # re-run even if outputs already exist
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
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
IMAGES_DIR   = PROJECT_ROOT / "Images"
OUT_MEDIAN   = PROJECT_ROOT / "output" / "median"
OUT_ADAPTIVE = PROJECT_ROOT / "output" / "adaptive-median"
OUT_WIENER   = PROJECT_ROOT / "output" / "wiener"
OUT_REPORT   = PROJECT_ROOT / "output" / "report"

# BrainWeb images (T1, 1mm, RF20%)
IMAGES = [
    "t1_icbm_normal_1mm_pn3_rf20.nii.gz",
    "t1_icbm_normal_1mm_pn5_rf20.nii.gz",
    "t1_icbm_normal_1mm_pn9_rf20.nii.gz",
]

NOISE_LABELS = ["pn3 (3%)", "pn5 (5%)", "pn9 (9%)"]

# ---------------------------------------------------------------------------
# ITK non-ASCII path workaround
# ---------------------------------------------------------------------------
_TMP_DIR: Path | None = None


def _itk_safe(p: Path) -> str:
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
    """Load a NIfTI volume as a uint8 numpy array (Z, Y, X)."""
    ImageType = itk.Image[itk.UC, 3]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(_itk_safe(path))
    reader.Update()
    return itk.array_from_image(reader.GetOutput())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stem(p: Path) -> str:
    n = p.name
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return p.stem


def three_views(arr: np.ndarray):
    """Return (axial, sagittal, coronal) middle slices."""
    z, y, x = arr.shape
    return arr[z // 2, :, :], arr[:, :, x // 2], arr[:, y // 2, :]


def run_script(script: str, args: list[str], label: str) -> int:
    cmd = [sys.executable, str(SRC_DIR / script)] + args
    print(f"    running {label} ...", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    ERROR in {label}:\n{r.stderr or r.stdout}", file=sys.stderr)
    return r.returncode


# ---------------------------------------------------------------------------
# Per-image comparison PNG  (4 cols x 3 rows)
# ---------------------------------------------------------------------------
def make_comparison_png(
    base: str,
    arrs: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    col_titles = list(arrs.keys())
    view_labels = ["Axial", "Sagittal", "Coronal"]
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
# Summary mosaic  (noise levels as rows, filters as cols, axial view only)
# ---------------------------------------------------------------------------
def make_summary_png(
    all_data: list[dict[str, np.ndarray]],
    noise_labels: list[str],
    out_path: Path,
) -> None:
    col_titles = list(all_data[0].keys())
    n_rows = len(all_data)
    n_cols = len(col_titles)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows)
    )
    # ensure 2-D axes array even with 1 row
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (data, noise_lbl) in enumerate(zip(all_data, noise_labels)):
        for col, title in enumerate(col_titles):
            ax = axes[row, col]
            axial = three_views(data[title])[0]
            ax.imshow(axial, cmap="gray", interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(noise_lbl, fontsize=9)
                ax.yaxis.set_visible(True)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_yticks([])

    fig.suptitle("All filters x all noise levels (axial slice)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SUMMARY] {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all filters on all BrainWeb images and produce report PNGs."
    )
    parser.add_argument("--median-radius", type=int, default=1,
                        help="Radius for ITK MedianImageFilter (default 1)")
    parser.add_argument("--max-window",    type=int, default=7,
                        help="Max window for adaptive median (default 7)")
    parser.add_argument("--wiener-window", type=int, default=5,
                        help="Window size for adaptive Wiener (default 5)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run filters even if output already exists")
    args = parser.parse_args()

    r  = args.median_radius
    mw = args.max_window
    ww = args.wiener_window

    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    OUT_MEDIAN.mkdir(parents=True, exist_ok=True)
    OUT_ADAPTIVE.mkdir(parents=True, exist_ok=True)
    OUT_WIENER.mkdir(parents=True, exist_ok=True)

    all_data: list[dict[str, np.ndarray]] = []

    for img_name, noise_lbl in zip(IMAGES, NOISE_LABELS):
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            print(f"[SKIP] {img_name} not found in Images/", flush=True)
            continue

        b = stem(img_path)
        print(f"\n=== {noise_lbl}  ({img_name}) ===", flush=True)

        # --- Median ---
        med_out = OUT_MEDIAN / f"{b}_median_r{r}.nii"
        if args.force or not med_out.exists():
            rc = run_script("median.py",
                            [str(img_path), str(r), str(med_out)],
                            f"median r={r}")
            if rc != 0:
                sys.exit(rc)
        else:
            print(f"    [skip] median already exists", flush=True)

        # --- Adaptive Median (full 3D NumPy paper algorithm) ---
        adap_out = OUT_ADAPTIVE / f"{b}_adaptive_median.nii"
        if args.force or not adap_out.exists():
            rc = run_script("adaptive-median.py",
                            [str(img_path), str(adap_out),
                             "--max-window", str(mw), "--no-itk"],
                            f"adaptive median mw={mw} (NumPy)")
            if rc != 0:
                sys.exit(rc)
        else:
            print(f"    [skip] adaptive median already exists", flush=True)

        # --- Adaptive Wiener ---
        wien_out = OUT_WIENER / f"{b}_wiener.nii"
        if args.force or not wien_out.exists():
            rc = run_script("wiener.py",
                            [str(img_path), str(wien_out),
                             "--window", str(ww)],
                            f"wiener w={ww}")
            if rc != 0:
                sys.exit(rc)
        else:
            print(f"    [skip] wiener already exists", flush=True)

        # --- Load all 4 volumes ---
        print(f"    loading volumes ...", flush=True)
        arrs = {
            "Original":        _load(img_path),
            f"Median (r={r})": _load(med_out),
            f"Adap. Median":   _load(adap_out),
            f"Adap. Wiener":   _load(wien_out),
        }

        # --- Per-image comparison PNG ---
        png_out = OUT_REPORT / f"{b}_comparison.png"
        make_comparison_png(f"{noise_lbl} - {b}", arrs, png_out)

        all_data.append(arrs)

    # --- Summary mosaic ---
    if all_data:
        present_labels = [
            lbl for lbl, img in zip(NOISE_LABELS, IMAGES)
            if (IMAGES_DIR / img).exists()
        ]
        summary_out = OUT_REPORT / "summary_axial.png"
        make_summary_png(all_data, present_labels, summary_out)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if _TMP_DIR is not None and _TMP_DIR.exists():
            shutil.rmtree(_TMP_DIR, ignore_errors=True)
