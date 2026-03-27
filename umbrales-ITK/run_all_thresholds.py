#!/usr/bin/env python3
"""
Procesa todos los volúmenes en umbrales-ITK/samples con Otsu, Huang y Triangle;
guarda NIfTI en umbrales-ITK/output y PNG de comparación en output/comparisons.

Cada filtro usa SetNumberOfHistogramBins (número de barras del histograma interno).
Por defecto se ejecutan varias configuraciones (32, 64, 128, 256); use --bins para
acotar o fijar un solo valor (un solo N → nombres sin sufijo _bN).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import itk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLES = BASE_DIR / "samples"
DEFAULT_OUTPUT = BASE_DIR / "output"
# Barras del histograma por defecto (una corrida por valor; sufijos _b32, _b64, …)
DEFAULT_HISTOGRAM_BINS: tuple[int, ...] = (32, 64, 128, 256)

ImageType = itk.Image[itk.US, 3]


def list_sample_volumes(samples_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(samples_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".nii.gz") or name.endswith(".nii"):
            files.append(p)
    return files


def load_as_uint16_3d(path: Path) -> itk.Image:
    img = itk.imread(str(path))
    arr = np.asarray(itk.array_from_image(img))
    if arr.ndim != 3:
        raise ValueError(f"Se espera volumen 3D; {path.name} tiene forma {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer) and arr.max() <= 65535 and arr.min() >= 0:
        arr_u = arr.astype(np.uint16)
    else:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            arr_u = np.zeros(arr.shape, dtype=np.uint16)
        else:
            arr_u = ((arr.astype(np.float64) - lo) / (hi - lo) * 65535.0).astype(np.uint16)

    out = itk.image_from_array(np.ascontiguousarray(arr_u))
    out.CopyInformation(img)
    return out


def threshold_and_rescale(
    img: itk.Image,
    filter_cls,
    number_of_bins: int,
) -> itk.Image:
    filter_type = filter_cls[ImageType, ImageType]
    flt = filter_type.New()
    flt.SetInput(img)
    flt.SetNumberOfHistogramBins(number_of_bins)
    flt.Update()

    rescaler = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
    rescaler.SetInput(flt.GetOutput())
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    rescaler.Update()
    return rescaler.GetOutput()


def middle_axial_slice(arr: np.ndarray) -> np.ndarray:
    z = arr.shape[0] // 2
    return np.asarray(arr[z, :, :], dtype=np.float64)


def save_comparison_png(
    original: np.ndarray,
    otsu: np.ndarray,
    huang: np.ndarray,
    triangle: np.ndarray,
    out_path: Path,
    title_stem: str,
    histogram_bins: int,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    panels = [
        (original, "Original"),
        (otsu, "Otsu"),
        (huang, "Huang"),
        (triangle, "Triangle"),
    ]
    for ax, (sl, label) in zip(axes, panels):
        ax.imshow(sl, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(f"{title_stem} · histograma: {histogram_bins} barras")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_one(
    input_path: Path,
    output_dir: Path,
    comparisons_dir: Path,
    number_of_bins: int,
    file_suffix: str,
) -> None:
    stem = input_path.name
    if stem.lower().endswith(".nii.gz"):
        base = stem[:-7]
    elif stem.lower().endswith(".nii"):
        base = stem[:-4]
    else:
        base = input_path.stem

    img_us = load_as_uint16_3d(input_path)

    otsu_out = threshold_and_rescale(
        img_us, itk.OtsuThresholdImageFilter, number_of_bins
    )
    huang_out = threshold_and_rescale(
        img_us, itk.HuangThresholdImageFilter, number_of_bins
    )
    triangle_out = threshold_and_rescale(
        img_us, itk.TriangleThresholdImageFilter, number_of_bins
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    itk.imwrite(otsu_out, str(output_dir / f"{base}_otsu{file_suffix}.nii"))
    itk.imwrite(huang_out, str(output_dir / f"{base}_huang{file_suffix}.nii"))
    itk.imwrite(triangle_out, str(output_dir / f"{base}_triangle{file_suffix}.nii"))

    orig_arr = itk.array_from_image(img_us)
    png_name = f"{base}_comparison{file_suffix}.png"
    save_comparison_png(
        middle_axial_slice(orig_arr),
        middle_axial_slice(itk.array_from_image(otsu_out)),
        middle_axial_slice(itk.array_from_image(huang_out)),
        middle_axial_slice(itk.array_from_image(triangle_out)),
        comparisons_dir / png_name,
        base,
        number_of_bins,
    )
    print(
        f"OK: {input_path.name} · {number_of_bins} barras · "
        f"comparisons/{png_name}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Umbrales Otsu/Huang/Triangle sobre todos los volúmenes en samples/."
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES,
        help="Carpeta con .nii / .nii.gz (por defecto: umbrales-ITK/samples)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Salida NIfTI y subcarpeta comparisons/ (por defecto: umbrales-ITK/output)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        nargs="+",
        default=list(DEFAULT_HISTOGRAM_BINS),
        metavar="N",
        help=(
            "Barras del histograma (SetNumberOfHistogramBins). Por defecto: "
            f"{', '.join(str(b) for b in DEFAULT_HISTOGRAM_BINS)} (salidas con sufijo _b<N>). "
            "Un solo valor: sin sufijo en nombres. Ej. solo 128: --bins 128"
        ),
    )
    args = parser.parse_args()

    samples_dir = args.samples_dir.resolve()
    output_dir = args.output_dir.resolve()
    comparisons_dir = output_dir / "comparisons"
    multi_bins = len(args.bins) > 1

    if not samples_dir.is_dir():
        print(f"No existe la carpeta de muestras: {samples_dir}", file=sys.stderr)
        return 1

    volumes = list_sample_volumes(samples_dir)
    if not volumes:
        print(f"No hay volúmenes (.nii / .nii.gz) en {samples_dir}", file=sys.stderr)
        return 1

    for path in volumes:
        for nbins in args.bins:
            suffix = f"_b{nbins}" if multi_bins else ""
            try:
                process_one(path, output_dir, comparisons_dir, nbins, suffix)
            except Exception as exc:  # noqa: BLE001
                print(f"Error en {path} (bins={nbins}): {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
