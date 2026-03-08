"""
Generate PNG comparison panels from 3D filter results.
Loads volumes, extracts middle axial slice, and builds one-row comparison figures.
Does not modify any logic in src/.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import itk
import matplotlib.pyplot as plt
import numpy as np


def load_volume(path: Path):
    """Load a 3D volume from path (NIfTI or NIfTI.gz). Returns ITK image."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return itk.imread(str(path))


def extract_middle_axial_slice(volume) -> np.ndarray:
    """
    Extract the middle axial slice from a 3D volume.
    ITK size is (x, y, z); GetArrayFromImage gives shape (z, y, x).
    Middle axial = slice at z = shape[0] // 2.
    Returns 2D numpy array (y, x) for display.
    """
    arr = itk.array_from_image(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got ndim={arr.ndim}")
    mid = arr.shape[0] // 2
    return arr[mid, :, :]


def create_comparison_figure(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    figure_title: Optional[str] = None,
    *,
    dpi: int = 200,
) -> None:
    """
    Create a one-row multi-column figure and save as PNG.
    images: list of 2D arrays (grayscale).
    titles: one title per image.
    Hides axes and uses grayscale.
    """
    if len(images) != len(titles):
        raise ValueError("images and titles must have same length")
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    if figure_title:
        fig.suptitle(figure_title, fontsize=12, y=1.02)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_filter_comparisons(
    sample_path: Path,
    result_paths_by_label: dict[str, Path],
    output_path: Path,
    filter_label: str,
    *,
    dpi: int = 200,
) -> bool:
    """
    Build comparison figure: Original | label1 | label2 | ...
    result_paths_by_label: e.g. {"Original": sample_path, "r2": path_r2, "r3": path_r3, "r4": path_r4}.
    If any path is missing, print warning and return False (skip).
    Returns True if figure was saved.
    """
    labels = list(result_paths_by_label.keys())
    images = []
    for label in labels:
        path = result_paths_by_label[label]
        if not Path(path).is_file():
            print(f"    [comparison] WARNING: missing {path}, skipping comparison for {sample_path.name}")
            return False
        try:
            vol = load_volume(path)
            sl = extract_middle_axial_slice(vol)
            images.append(sl)
        except Exception as e:
            print(f"    [comparison] WARNING: failed to load {path}: {e}, skipping")
            return False
    titles = labels
    base = _basename(sample_path)
    out_file = output_path / f"{base}_comparison.png"
    create_comparison_figure(
        images,
        titles,
        out_file,
        figure_title=f"{filter_label}: {base}",
        dpi=dpi,
    )
    print(f"    [comparison] saved {out_file}")
    return True


def _basename(path: Path) -> str:
    """Sample base name without .nii / .nii.gz."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def generate_all_comparisons(
    samples_dir: Path,
    result_base: Path,
    comparison_base: Path,
    mean_radii: list[int],
    median_radii: list[int],
    *,
    histogram_combos: Optional[list[tuple[float, float, int]]] = None,
    dpi: int = 200,
) -> None:
    """
    For each sample, generate comparison PNGs for:
    - mean (Original | r2 | r3 | r4)
    - median (Original | r2 | r3 | r4)
    - gradient (Original | Gradient)
    - adaptive_histogram (Original | combo1 | combo2 | ...) if histogram_combos given
    """
    comparison_base = Path(comparison_base)
    result_base = Path(result_base)
    mean_out = comparison_base / "mean"
    median_out = comparison_base / "median"
    gradient_out = comparison_base / "gradient"
    hist_out = comparison_base / "adaptive_histogram"
    for d in (mean_out, median_out, gradient_out, hist_out):
        d.mkdir(parents=True, exist_ok=True)

    mean_dir = result_base / "mean_results"
    median_dir = result_base / "median_results"
    gradient_dir = result_base / "gradient_results"
    hist_dir = result_base / "adaptive_histogram_results"

    samples = _list_nifti(samples_dir)
    if not samples:
        print("[comparison] No sample images found, skipping comparisons.")
        return

    print("[comparison] Generating comparison figures (mean, median, gradient, adaptive_histogram)...")
    for sample_path in samples:
        base = _basename(sample_path)
        # Mean: Original | r2 | r3 | r4
        result_paths = {"Original": sample_path}
        for r in mean_radii:
            result_paths[f"r{r}"] = mean_dir / f"{base}_mean_r{r}.nii"
        generate_filter_comparisons(
            sample_path,
            result_paths,
            mean_out,
            "Mean",
            dpi=dpi,
        )
        # Median: Original | r2 | r3 | r4
        result_paths_med = {"Original": sample_path}
        for r in median_radii:
            result_paths_med[f"r{r}"] = median_dir / f"{base}_median_r{r}.nii"
        generate_filter_comparisons(
            sample_path,
            result_paths_med,
            median_out,
            "Median",
            dpi=dpi,
        )
        # Gradient: Original | Gradient (single output per sample)
        result_paths_grad = {"Original": sample_path, "Gradient": gradient_dir / f"{base}_gradient.nii"}
        generate_filter_comparisons(
            sample_path,
            result_paths_grad,
            gradient_out,
            "Gradient",
            dpi=dpi,
        )
        # Adaptive histogram: Original | combo1 | combo2 | ...
        if histogram_combos:
            result_paths_hist = {"Original": sample_path}
            for a, b, r in histogram_combos:
                label = f"a{a}_b{b}_r{r}"
                result_paths_hist[label] = hist_dir / f"{base}_hist_a{a}_b{b}_r{r}.nii"
            generate_filter_comparisons(
                sample_path,
                result_paths_hist,
                hist_out,
                "Adaptive histogram",
                dpi=dpi,
            )
    print("[comparison] Done.")


def _list_nifti(dir_path: Path) -> list[Path]:
    """List .nii and .nii.gz files in directory."""
    if not Path(dir_path).is_dir():
        return []
    out = []
    for p in sorted(Path(dir_path).iterdir()):
        if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii")):
            out.append(p)
    return out
