#!/usr/bin/env python
"""
Taller — sección 2.2.2: Adaptive median filtering.

Para esto hay que probar con varios porcentajes de ruidos (nota del taller).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import itk
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tqdm opcional — si no está instalado, el script sigue funcionando
try:
    from tqdm import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ITK path workaround: C++ cannot handle non-ASCII paths on Windows
_TMP_DIR = None


def _itk_safe(p: Path) -> str:
    """Copy file to ASCII-only temp dir if path has non-ASCII chars."""
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_adm_"))
        dst = _TMP_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        return str(dst)


def _itk_safe_write(p: Path) -> str:
    """Return ASCII-safe output path; caller must move the file if different."""
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_adm_"))
        return str(_TMP_DIR / p.name)


# Raíz del proyecto taller-class-filter-image (padre de src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "Images"
OUTPUT_ADAPTIVE_MEDIAN_DIR = PROJECT_ROOT / "output" / "adaptive-median"
OUTPUT_COMPARISON_DIR = PROJECT_ROOT / "output" / "comparison_results"

EXPERIMENT_COMBOS: list[tuple[str, float, float, int, str]] = [
    # (noise_type, noise_density, noise_sigma, max_window, label)
    ("none", 0.0, 0.0, 5, "no_noise_mw5"),
    ("none", 0.0, 0.0, 7, "no_noise_mw7"),
    ("none", 0.0, 0.0, 9, "no_noise_mw9"),
    ("salt_pepper", 0.05, 0.0, 7, "sp005_mw7"),
    ("salt_pepper", 0.1, 0.0, 7, "sp010_mw7"),
    ("salt_pepper", 0.3, 0.0, 7, "sp030_mw7"),
    ("salt_pepper", 0.1, 0.0, 5, "sp010_mw5"),
    ("salt_pepper", 0.1, 0.0, 9, "sp010_mw9"),
    ("gaussian", 0.0, 5.0, 7, "gs5_mw7"),
    ("gaussian", 0.0, 10.0, 7, "gs10_mw7"),
    ("gaussian", 0.0, 20.0, 7, "gs20_mw7"),
    ("mixed", 0.1, 10.0, 7, "mixed_mw7"),
]


def resolve_input_path(user_path: str) -> Path:
    """
    Si la ruta no existe como está, intenta Images/<nombre>.
    Permite: `Images/foo.nii.gz`, `foo.nii.gz` o ruta absoluta.
    """
    p = Path(user_path)
    if p.is_file():
        return p.resolve()
    alt = IMAGES_DIR / user_path
    if alt.is_file():
        return alt.resolve()
    print(f"ERROR: no se encontró la imagen de entrada: {user_path}", file=sys.stderr)
    print(f"  Buscado también en: {alt}", file=sys.stderr)
    sys.exit(1)


def default_output_path(input_path: Path) -> Path:
    """Salida por defecto: output/adaptive-median/<base>_adaptive_median.nii"""
    name = input_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = input_path.stem
    OUTPUT_ADAPTIVE_MEDIAN_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ADAPTIVE_MEDIAN_DIR / f"{stem}_adaptive_median.nii"


def _run_itk(reader_output, max_window: int):
    """
    Aproximación con primitivas ITK (no es el algoritmo adaptativo exacto de Ali 2018).

    Se aplica MedianImageFilter con vecindario 3×3 en plano (radio 1 en x,y; 0 en z)
    para suavizar manteniendo un solo corte axial de grosor en z. Luego
    GradientMagnitudeImageFilter sobre el resultado mediano detecta bordes/transiciones.
    Donde el gradiente supera un umbral (percentil 82 de la magnitud), se restaura el
    voxel original: en regiones homogéneas predomina la mediana; en bordes se preserva
    la intensidad original. max_window se documenta para coherencia con la API pero no
    interviene en esta aproximación (el tamaño efectivo de mediana fija es 3×3 en plano).
    """
    _ = max_window  # reservado para coherencia con la API
    PixelType = itk.UC
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]

    orig_arr = itk.array_from_image(reader_output)

    median_filter = itk.MedianImageFilter[ImageType, ImageType].New()
    median_filter.SetInput(reader_output)
    radius = itk.Size[Dimension]()
    radius[0] = 1
    radius[1] = 1
    radius[2] = 0
    median_filter.SetRadius(radius)
    median_filter.Update()
    med_img = median_filter.GetOutput()

    grad_filter = itk.GradientMagnitudeImageFilter[ImageType, ImageType].New()
    grad_filter.SetInput(med_img)
    grad_filter.Update()
    grad_img = grad_filter.GetOutput()

    med_arr = itk.array_from_image(med_img)
    grad_arr = itk.array_from_image(grad_img).astype(np.float64)
    thresh = float(np.percentile(grad_arr, 82.0))
    mask = grad_arr > thresh
    out_arr = np.where(mask, orig_arr, med_arr).astype(np.uint8)

    out_img = itk.image_from_array(out_arr)
    out_img.CopyInformation(reader_output)
    return out_img


def _adaptive_median_slice_2d(slice_2d: np.ndarray, smax: int) -> np.ndarray:
    """Un corte axial: algoritmo adaptativo (Ali 2018, sección 2.2.2)."""
    h, w = slice_2d.shape
    pad = smax // 2 + 1
    padded = np.pad(slice_2d.astype(np.uint8), pad, mode="reflect")
    out = np.zeros((h, w), dtype=np.uint8)
    for yi in range(h):
        for xi in range(w):
            py, px = yi + pad, xi + pad
            win = 3
            zxy = int(slice_2d[yi, xi])
            while True:
                r = win // 2
                neigh = padded[py - r : py + r + 1, px - r : px + r + 1]
                zmin = int(neigh.min())
                zmax = int(neigh.max())
                zmed = int(np.median(neigh))
                la1 = zmed - zmin
                la2 = zmed - zmax
                if la1 > 0 and la2 < 0:
                    lb1 = zxy - zmin
                    lb2 = zxy - zmax
                    if lb1 > 0 and lb2 < 0:
                        out[yi, xi] = zxy
                    else:
                        out[yi, xi] = zmed
                    break
                win += 2
                if win > smax:
                    out[yi, xi] = zmed
                    break
    return out


def _run_numpy(reader_output, max_window: int):
    """Implementación pura NumPy del filtro mediana adaptativo (Ali 2018), corte a corte axial."""
    arr = itk.array_from_image(reader_output)
    smax = max_window if max_window % 2 == 1 else max_window - 1
    if smax < 3:
        smax = 3
    out_vol = np.zeros_like(arr, dtype=np.uint8)
    for z in range(arr.shape[0]):
        out_vol[z] = _adaptive_median_slice_2d(arr[z], smax)
    out_img = itk.image_from_array(out_vol)
    out_img.CopyInformation(reader_output)
    return out_img


def inject_salt_pepper(arr: np.ndarray, density: float) -> np.ndarray:
    """
    Añade ruido sal y pimienta: `density` fracción del total de vóxeles.
    Mitad se pone a 255 (sal), mitad a 0 (pimienta). Trabaja sobre una copia.
    """
    out = arr.copy().astype(np.uint8)
    total = out.size
    n_noise = int(total * density)
    if n_noise <= 0:
        return out
    indices = np.random.choice(total, n_noise, replace=False)
    half = n_noise // 2
    out.flat[indices[:half]] = 255
    out.flat[indices[half:]] = 0
    return out


def inject_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Añade ruido Gaussiano N(0, sigma). Recorta a [0,255] y convierte a uint8.
    """
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def inject_mixed(arr: np.ndarray, density: float, sigma: float) -> np.ndarray:
    """Aplica ruido Gaussiano primero y luego sal y pimienta encima."""
    return inject_salt_pepper(inject_gaussian(arr, sigma), density)


def apply_noise(
    arr: np.ndarray, noise_type: str, density: float, sigma: float
) -> np.ndarray:
    """Despachador: llama la función de ruido correspondiente."""
    if noise_type == "salt_pepper":
        return inject_salt_pepper(arr, density)
    if noise_type == "gaussian":
        return inject_gaussian(arr, sigma)
    if noise_type == "mixed":
        return inject_mixed(arr, density, sigma)
    return arr.copy()  # 'none'


def _get_stem(input_path: Path) -> str:
    """Nombre base del archivo sin extensión NIfTI."""
    name = input_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return input_path.stem


def _middle_slice(arr: np.ndarray) -> np.ndarray:
    """Corte axial central de un volumen 3D (shape: z,y,x)."""
    return arr[arr.shape[0] // 2, :, :]


def _three_views(arr: np.ndarray) -> tuple:
    """Axial, sagittal, coronal middle slices from a (Z,Y,X) volume."""
    z, y, x = arr.shape
    return arr[z // 2, :, :], arr[:, :, x // 2], arr[:, y // 2, :]


def _arr_to_itk(arr: np.ndarray, reference) -> itk.Image:
    """Convierte array numpy uint8 a itk.Image[UC,3] copiando metadatos."""
    img = itk.image_from_array(arr.astype(np.uint8))
    img.CopyInformation(reference)
    return img


def _apply_filter(noisy_itk, smax: int, use_numpy: bool) -> itk.Image:
    """Llama _run_numpy o _run_itk según use_numpy."""
    if use_numpy:
        return _run_numpy(noisy_itk, smax)
    return _run_itk(noisy_itk, smax)


def save_single_comparison(
    orig_arr: np.ndarray,
    noisy_arr: np.ndarray,
    filtered_arr: np.ndarray,
    title: str,
    out_path: Path,
    noise_type: str = "none",
) -> None:
    """
    Guarda figura PNG con 3 vistas anatomicas (axial, sagittal, coronal):
      - Sin ruido: 2 columnas  -> Original | Adaptive Median
      - Con ruido: 3 columnas  -> Original | Ruidosa | Adaptive Median
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if noise_type == "none":
        volumes = [orig_arr, filtered_arr]
        col_titles = ["Original", "Adaptive Median"]
    else:
        volumes = [orig_arr, noisy_arr, filtered_arr]
        col_titles = ["Original", f"Ruido ({noise_type})", "Adaptive Median"]

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


def _save_experiment_summary(
    base: str,
    orig_arr: np.ndarray,
    results: dict[str, dict],
) -> None:
    """
    Genera una figura resumen de 4 filas con todas las combinaciones.

    results: dict  label → {"noisy_arr": np.ndarray, "filtered_arr": np.ndarray}

    Layout (cada celda = corte axial central):
      Fila 1 — Efecto max_window (sin ruido):
          Original | mw=5 | mw=7 | mw=9
      Fila 2 — Efecto densidad sal y pimienta (mw=7):
          orig+sp005|filt | orig+sp010|filt | orig+sp030|filt
      Fila 3 — Efecto sigma gaussiano (mw=7):
          orig+gs5|filt | orig+gs10|filt | orig+gs20|filt
      Fila 4 — Ruido mixto:
          Original | orig+mixed | filtrada

    Usa fig.text() para el título de cada fila en el margen izquierdo.
    Título general: "<base> — Adaptive Median Filter: Análisis de Parámetros"
    Guarda en: output/comparison_results/<base>_experiment_summary.png
    dpi=200, bbox_inches='tight'
    """
    OUTPUT_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_COMPARISON_DIR / f"{base}_experiment_summary.png"
    orig_sl = _middle_slice(orig_arr)

    # Grid: 4 filas × 6 columnas (máximo de columnas necesarias)
    n_rows, n_cols = 4, 6
    fig = plt.figure(figsize=(3.0 * n_cols, 3.5 * n_rows))

    def place(row: int, col: int) -> plt.Axes:
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
        ax.axis("off")
        return ax

    def show(ax, arr2d, title=""):
        ax.imshow(arr2d, cmap="gray")
        ax.set_title(title, fontsize=7)

    # --- Fila 0: efecto max_window ---
    fig.text(
        0.005,
        0.87,
        "Efecto\nmax_window\n(sin ruido)",
        va="center",
        ha="left",
        fontsize=8,
        fontweight="bold",
        rotation=90,
    )
    show(place(0, 0), orig_sl, "Original")
    for i, lbl in enumerate(["no_noise_mw5", "no_noise_mw7", "no_noise_mw9"]):
        if lbl in results:
            mw = lbl.split("mw")[-1]
            show(place(0, i + 1), _middle_slice(results[lbl]["filtered_arr"]), f"mw={mw}")

    # --- Fila 1: efecto densidad sal y pimienta ---
    fig.text(
        0.005,
        0.625,
        "Efecto\nsal y pimienta\n(mw=7)",
        va="center",
        ha="left",
        fontsize=8,
        fontweight="bold",
        rotation=90,
    )
    col = 0
    for lbl, dens_label in [
        ("sp005_mw7", "sp=0.05"),
        ("sp010_mw7", "sp=0.10"),
        ("sp030_mw7", "sp=0.30"),
    ]:
        if lbl in results and col + 1 < n_cols:
            show(place(1, col), _middle_slice(results[lbl]["noisy_arr"]), f"Ruido {dens_label}")
            show(
                place(1, col + 1),
                _middle_slice(results[lbl]["filtered_arr"]),
                f"Filtrada {dens_label}",
            )
            col += 2

    # --- Fila 2: efecto sigma gaussiano ---
    fig.text(
        0.005,
        0.375,
        "Efecto\nruido Gauss.\n(mw=7)",
        va="center",
        ha="left",
        fontsize=8,
        fontweight="bold",
        rotation=90,
    )
    col = 0
    for lbl, gs_label in [
        ("gs5_mw7", "σ=5"),
        ("gs10_mw7", "σ=10"),
        ("gs20_mw7", "σ=20"),
    ]:
        if lbl in results and col + 1 < n_cols:
            show(place(2, col), _middle_slice(results[lbl]["noisy_arr"]), f"Ruido {gs_label}")
            show(
                place(2, col + 1),
                _middle_slice(results[lbl]["filtered_arr"]),
                f"Filtrada {gs_label}",
            )
            col += 2

    # --- Fila 3: ruido mixto ---
    fig.text(
        0.005,
        0.125,
        "Ruido mixto\n(sp+Gauss, mw=7)",
        va="center",
        ha="left",
        fontsize=8,
        fontweight="bold",
        rotation=90,
    )
    lbl = "mixed_mw7"
    if lbl in results:
        show(place(3, 0), orig_sl, "Original")
        show(place(3, 1), _middle_slice(results[lbl]["noisy_arr"]), "Ruido mixto")
        show(place(3, 2), _middle_slice(results[lbl]["filtered_arr"]), "Filtrada")

    fig.suptitle(
        f"{base} — Adaptive Median Filter: Análisis de Parámetros",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [resumen] {out_path}")


def run_experiment(input_path: Path, reader_output, use_numpy: bool) -> None:
    """
    Ejecuta todas las combinaciones de EXPERIMENT_COMBOS.

    Para cada combinación:
      1. Obtiene array original del reader_output.
      2. Aplica apply_noise().
      3. Convierte array ruidoso a itk.Image con _arr_to_itk().
      4. Aplica el filtro con _apply_filter().
      5. Guarda NIfTI en output/adaptive-median/<base>_<label>_adaptive_median.nii
      6. Guarda PNG individual en output/comparison_results/<base>_<label>_comparison.png
      7. Imprime progreso: "[N/12] label — guardado"

    Al finalizar llama _save_experiment_summary() con todos los resultados.
    """
    base = _get_stem(input_path)
    orig_arr = itk.array_from_image(reader_output)
    total = len(EXPERIMENT_COMBOS)
    results: dict[str, dict] = {}

    OUTPUT_ADAPTIVE_MEDIAN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    combo_iter = EXPERIMENT_COMBOS
    if _TQDM_AVAILABLE:
        combo_iter = _tqdm(EXPERIMENT_COMBOS, desc="Experimentos", unit="combo")

    for i, (noise_type, density, sigma, mw, label) in enumerate(combo_iter, 1):
        if not _TQDM_AVAILABLE:
            print(f"[{i}/{total}] {label} ...", flush=True)

        # 1. Ruido
        noisy_arr = apply_noise(orig_arr, noise_type, density, sigma)

        # 2. Filtro
        noisy_itk = _arr_to_itk(noisy_arr, reader_output)
        filtered_itk = _apply_filter(noisy_itk, mw, use_numpy)
        filtered_arr = itk.array_from_image(filtered_itk)

        # 3. Guardar NIfTI
        nii_path = OUTPUT_ADAPTIVE_MEDIAN_DIR / f"{base}_{label}_adaptive_median.nii"
        safe_nii = _itk_safe_write(nii_path)
        writer = itk.ImageFileWriter[itk.Image[itk.UC, 3]].New()
        writer.SetFileName(safe_nii)
        writer.SetInput(filtered_itk)
        writer.Update()
        if safe_nii != str(nii_path):
            nii_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(safe_nii, nii_path)

        # 4. Guardar PNG individual
        png_path = OUTPUT_COMPARISON_DIR / f"{base}_{label}_comparison.png"
        save_single_comparison(
            orig_arr,
            noisy_arr,
            filtered_arr,
            title=f"{base} — {label}",
            out_path=png_path,
            noise_type=noise_type,
        )

        # 5. Acumular para resumen
        results[label] = {"noisy_arr": noisy_arr, "filtered_arr": filtered_arr}
        print(f"  -> {nii_path.name}", flush=True)

    # Figura resumen global
    _save_experiment_summary(base, orig_arr, results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive median filtering — 2.2.2 (taller)."
    )
    parser.add_argument(
        "input_image",
        help="Ruta al volumen de entrada o nombre de archivo dentro de Images/",
    )
    parser.add_argument(
        "output_image",
        nargs="?",
        default=None,
        help=(
            "Ruta de salida (.nii recomendado). "
            "Si se omite, se usa output/adaptive-median/<base>_adaptive_median.nii"
        ),
    )
    parser.add_argument(
        "--max-window",
        type=int,
        default=None,
        help="Smax: tamaño máximo de ventana impar (por defecto 7).",
    )
    parser.add_argument(
        "--no-itk",
        action="store_true",
        help="Si se indica, usa implementación NumPy; si no, aproximación ITK (por defecto).",
    )
    parser.add_argument(
        "--noise-type",
        choices=["none", "salt_pepper", "gaussian", "mixed"],
        default="none",
        help="Tipo de ruido a inyectar antes de filtrar (por defecto 'none').",
    )
    parser.add_argument(
        "--noise-density",
        type=float,
        default=0.1,
        help="Fracción de píxeles afectados por sal y pimienta (por defecto 0.1).",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=10.0,
        help="Desviación estándar del ruido Gaussiano (por defecto 10.0).",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Ejecuta todas las combinaciones de parámetros automáticamente.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_image)
    if args.output_image is not None:
        out_path = Path(args.output_image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = default_output_path(input_path)

    reader = itk.ImageFileReader[itk.Image[itk.UC, 3]].New()
    reader.SetFileName(_itk_safe(input_path))
    reader.Update()
    smax = args.max_window if args.max_window is not None else 7
    if smax < 3:
        smax = 7

    # --- TALLER: 2.2.2 Adaptive median filtering ---
    if args.experiment:
        # Modo experimento: corre las 12 combinaciones y genera figuras resumen
        run_experiment(input_path, reader.GetOutput(), use_numpy=args.no_itk)
    else:
        # Modo normal: aplica ruido opcional + filtro + guarda NIfTI + PNG individual
        orig_arr = itk.array_from_image(reader.GetOutput())
        noisy_arr = apply_noise(
            orig_arr, args.noise_type, args.noise_density, args.noise_sigma
        )

        if np.array_equal(orig_arr, noisy_arr):
            noisy_input = reader.GetOutput()
        else:
            noisy_input = _arr_to_itk(noisy_arr, reader.GetOutput())

        result = _apply_filter(noisy_input, smax, use_numpy=args.no_itk)

        safe_out = _itk_safe_write(out_path)
        writer = itk.ImageFileWriter[itk.Image[itk.UC, 3]].New()
        writer.SetFileName(safe_out)
        writer.SetInput(result)
        writer.Update()
        if safe_out != str(out_path):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(safe_out, out_path)

        # Figura de comparación individual
        png_path = OUTPUT_COMPARISON_DIR / f"{_get_stem(input_path)}_single_comparison.png"
        save_single_comparison(
            orig_arr,
            noisy_arr,
            itk.array_from_image(result),
            title=f"{_get_stem(input_path)} — noise={args.noise_type} mw={smax}",
            out_path=png_path,
            noise_type=args.noise_type,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        if _TMP_DIR is not None and Path(_TMP_DIR).exists():
            shutil.rmtree(_TMP_DIR, ignore_errors=True)
