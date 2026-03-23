#!/usr/bin/env python
"""
Filtro de Wiener Adaptativo 3D — sección 2.1 del taller.

Basado en:
    Hanafy M. Ali, "MRI Medical Image Denoising by Fundamental Filters",
    Capítulo 7 en High-Resolution Neuroimaging - Basic Physical Principles
    and Clinical Applications, IntechOpen, 2018.
    DOI: 10.5772/intechopen.72427

Algoritmo (por vóxel):
    f_hat = mu + max(0, var_local - nu2) / max(var_local, eps) * (g - mu)

    mu        -- media local en una ventana cúbica de lado `window`
    var_local -- varianza local (E[X²] - E[X]²) en la misma ventana
    nu2       -- varianza global del ruido (promedio de todas las var. locales,
                 o suministrada por el usuario)
    g         -- vóxel ruidoso de entrada
    eps       -- épsilon de máquina (evita división por cero)
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

# Rutas del proyecto (misma convención que median.py / adaptive-median.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "Images"
OUTPUT_WIENER_DIR = PROJECT_ROOT / "output" / "wiener"
OUTPUT_COMPARISON_DIR = PROJECT_ROOT / "output" / "comparison_results"

# Tipo ITK para volúmenes float32
DIMENSION = 3
PIXEL_TYPE = itk.F
IMAGE_TYPE = itk.Image[PIXEL_TYPE, DIMENSION]


# -- Resolución de rutas -------------------------------------------------------

def resolve_input_path(user_path: str) -> Path:
    """
    Resuelve la ruta de entrada: acepta ruta absoluta o nombre de archivo
    dentro de Images/. Termina el proceso con error si no encuentra el archivo.
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


def _get_stem(p: Path) -> str:
    """Extrae el nombre base del archivo sin la extensión NIfTI (.nii o .nii.gz)."""
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def default_output_path(input_path: Path) -> Path:
    """Ruta de salida por defecto: output/wiener/<base>_wiener.nii"""
    OUTPUT_WIENER_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_WIENER_DIR / f"{_get_stem(input_path)}_wiener.nii"


# -- Manejo de rutas no-ASCII para ITK ----------------------------------------

# Directorio temporal para rutas no-ASCII (limitación C++ de ITK en Windows)
_TMP_DIR: Optional[Path] = None


def _itk_safe_path(p: Path) -> str:
    """
    La capa C++ de ITK no puede manejar caracteres no-ASCII en rutas de Windows.
    Si la ruta contiene tildes, ñ u otros caracteres no-ASCII, copia el archivo
    a un directorio temporal con ruta ASCII pura, conservando el nombre original
    (para que ITK detecte el formato por la extensión, ej: '.nii.gz').
    Devuelve la ruta segura como cadena.
    """
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s  # la ruta ya es ASCII-segura
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_wiener_"))
        dst = _TMP_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        return str(dst)


def _itk_safe_output(p: Path) -> str:
    """
    Devuelve una ruta de escritura ASCII-segura para ITK.
    Si el destino real tiene caracteres no-ASCII, retorna una ruta temporal;
    el llamador debe mover el archivo al destino real después de escribir.
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
    """
    Lee cualquier imagen médica 3D directamente como float32. Usa itk.imread
    con tipo itk.F para que ITK convierta automáticamente cualquier tipo nativo
    del archivo (uint8, uint16, float32, etc.) a float32, preservando el rango
    completo de intensidades sin recorte. Imprime tamaño y espaciado en consola.
    """
    safe = _itk_safe_path(Path(filepath))
    img = itk.imread(safe, itk.F)
    size = list(img.GetLargestPossibleRegion().GetSize())
    spacing = [round(s, 4) for s in img.GetSpacing()]
    print(f"    Cargado : {filepath}")
    print(f"    Tamaño  : {size}  (X × Y × Z vóxeles)")
    print(f"    Espaciado: {spacing}  mm/vóxel")
    return img


def numpy_to_itk(arr: np.ndarray, reference: itk.Image) -> itk.Image:
    """
    Envuelve un array NumPy float32 de vuelta como itk.Image,
    copiando los metadatos espaciales (espaciado, origen, dirección) del volumen
    de referencia para que el NIfTI de salida sea correcto.
    """
    out = itk.image_from_array(arr.astype(np.float32))
    out.SetSpacing(reference.GetSpacing())
    out.SetOrigin(reference.GetOrigin())
    out.SetDirection(reference.GetDirection())
    return out


def save_image(itk_img: itk.Image, filepath: str) -> None:
    """
    Guarda una imagen ITK en disco. Maneja rutas no-ASCII escribiendo primero
    en un directorio temporal y moviendo el archivo al destino real.
    """
    real = Path(filepath)
    safe = _itk_safe_output(real)
    writer = itk.ImageFileWriter[IMAGE_TYPE].New()
    writer.SetFileName(safe)
    writer.SetInput(itk_img)
    writer.Update()
    # Si ITK escribió en ruta temporal, mover al destino real
    if safe != str(real):
        real.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(safe, real)
    print(f"    Guardado: {filepath}")


# -- Algoritmo principal -------------------------------------------------------

def adaptive_wiener_filter_3d(
    arr: np.ndarray,
    window_size: int = 5,
    noise_variance: Optional[float] = None,
) -> np.ndarray:
    """
    Aplica el Filtro de Wiener Adaptativo 3D (Ali 2018, Sección 2.1).

    La ganancia de Wiener varía de 0 a 1 según las estadísticas locales:
      - Regiones planas (var_local ≤ nu2): ganancia ≈ 0 → suavizado máximo.
      - Bordes / estructuras (var_local >> nu2): ganancia ≈ 1 → señal preservada.

    Parámetros
    ----------
    arr            : np.ndarray — volumen 3D (Z, Y, X), cualquier tipo flotante
    window_size    : int        — lado de la ventana cúbica (debe ser impar)
    noise_variance : float|None — nu2; si es None se estima como promedio de
                                  todas las varianzas locales del volumen

    Retorna
    -------
    np.ndarray — volumen filtrado, dtype float32, misma forma que arr
    """
    arr_f = arr.astype(np.float64)

    # Paso 1 — media local mu (filtro de caja deslizante)
    local_mean = uniform_filter(arr_f, size=window_size, mode="reflect")

    # Paso 2 — varianza local = E[X²] - E[X]²
    local_mean_sq = uniform_filter(arr_f ** 2, size=window_size, mode="reflect")
    local_var = local_mean_sq - local_mean ** 2

    # Paso 3 — varianza global del ruido nu2 (estimada si no se proporcionó)
    if noise_variance is None:
        noise_variance = float(np.mean(local_var))
        print(f"    nu2 auto-estimada = {noise_variance:.4f}")
    else:
        print(f"    nu2 suministrada  = {noise_variance:.4f}")

    # Paso 4 — ganancia de Wiener por vóxel, acotada a [0, 1]
    eps = np.finfo(np.float64).eps
    gain = np.maximum(local_var - noise_variance, 0.0) / np.maximum(local_var, eps)

    # Paso 5 — restauración: f_hat = mu + ganancia × (g - mu)
    filtered = local_mean + gain * (arr_f - local_mean)
    return filtered.astype(np.float32)


# -- Métrica PSNR (sección 4 del artículo) ------------------------------------

def compute_psnr(reference: np.ndarray, filtered: np.ndarray,
                 max_val: Optional[float] = None) -> float:
    """
    Calcula el PSNR (Peak Signal-to-Noise Ratio) en decibelios entre la imagen
    de referencia y la imagen filtrada.
    PSNR = 10 × log10(MAX² / MSE)
    """
    if max_val is None:
        max_val = float(np.max(reference))
    mse = float(np.mean(
        (reference.astype(np.float64) - filtered.astype(np.float64)) ** 2
    ))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


# -- Inyección de ruido (para experimentos) ------------------------------------

def inject_noise(arr: np.ndarray, noise_type: str,
                 density: float, sigma: float) -> np.ndarray:
    """
    Inyecta ruido sintético sobre el volumen antes de filtrar.

    Tipos soportados:
      - 'salt_pepper': vóxeles aleatorios se ponen a MAX o MIN (densidad controlada).
      - 'gaussian'   : se suma ruido gaussiano N(0, sigma) a cada vóxel.
      - 'mixed'      : gaussiano seguido de sal y pimienta.
      - 'none'       : devuelve copia sin modificar.
    """
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
        # Primero gaussiano, luego sal y pimienta encima
        tmp = arr + np.random.normal(0.0, sigma, arr.shape).astype(arr.dtype)
        n = int(tmp.size * density)
        if n > 0:
            idx = np.random.choice(tmp.size, n, replace=False)
            tmp.flat[idx[: n // 2]] = float(arr.max())
            tmp.flat[idx[n // 2 :]] = float(arr.min())
        return tmp
    return arr.copy()  # 'none': sin ruido


# -- Generación de PNG comparativo ---------------------------------------------

def _three_views(arr: np.ndarray) -> tuple:
    """Devuelve los cortes centrales axial, sagital y coronal de un volumen (Z,Y,X)."""
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
    Guarda un PNG con los tres cortes anatómicos (axial, sagital, coronal):
      - Sin ruido: 2 columnas  → Original | Wiener (AWF)
      - Con ruido: 3 columnas  → Original | Ruidosa | Wiener (AWF)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if noise_type == "none":
        volumes = [orig_arr, filtered_arr]
        col_titles = ["Original", "Wiener (AWF)"]
    else:
        volumes = [orig_arr, noisy_arr, filtered_arr]
        col_titles = ["Original", f"Ruido ({noise_type})", "Wiener (AWF)"]

    n_cols = len(volumes)
    view_labels = ["Axial", "Sagital", "Coronal"]
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


# -- Interfaz de línea de comandos ---------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filtro de Wiener Adaptativo 3D (Ali 2018, sección 2.1). "
            "Entrada: volumen NIfTI; salida: NIfTI filtrado + PNG de comparación."
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
        help="Lado de la ventana cúbica N×N×N (impar; por defecto 5)",
    )
    parser.add_argument(
        "--noise-var",
        type=float,
        default=None,
        metavar="V",
        help="Varianza de ruido nu2 (por defecto: estimada automáticamente)",
    )
    parser.add_argument(
        "--noise-type",
        choices=["none", "salt_pepper", "gaussian", "mixed"],
        default="none",
        help="Tipo de ruido sintético a inyectar antes de filtrar (por defecto 'none')",
    )
    parser.add_argument(
        "--noise-density",
        type=float,
        default=0.1,
        help="Fracción de píxeles afectados por sal y pimienta (por defecto 0.1)",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=10.0,
        help="Desviación estándar del ruido gaussiano (por defecto 10.0)",
    )
    parser.add_argument(
        "--reference",
        default=None,
        metavar="FILE",
        help="Imagen de referencia limpia para calcular PSNR (opcional)",
    )
    args = parser.parse_args()

    # Validar que la ventana sea impar y mayor que cero
    if args.window < 1:
        sys.exit("ERROR: --window debe ser >= 1")
    if args.window % 2 == 0:
        args.window += 1
        print(f"AVISO: --window ajustado a {args.window} (se requiere número impar)")

    input_path = resolve_input_path(args.input_image)
    out_path = (
        Path(args.output_image) if args.output_image else default_output_path(input_path)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFiltro de Wiener Adaptativo — {input_path.name}")
    print(f"  Ventana : {args.window}×{args.window}×{args.window} vóxeles")
    print(f"  Salida  : {out_path}\n")

    # Paso 1 — Cargar imagen como float32
    print("--- 1/5 Cargando imagen ---")
    itk_input = load_as_float(str(input_path))
    arr_in = itk.array_from_image(itk_input)

    # Paso 2 — Inyección de ruido sintético (opcional)
    print("--- 2/5 Inyectando ruido ---")
    noisy_arr = inject_noise(arr_in, args.noise_type, args.noise_density, args.noise_sigma)
    if args.noise_type == "none":
        print("    Sin ruido sintético")
    else:
        print(f"    Ruido '{args.noise_type}' aplicado")

    # Paso 3 — Aplicar filtro de Wiener adaptativo
    print("--- 3/5 Aplicando Wiener adaptativo ---")
    arr_out = adaptive_wiener_filter_3d(
        noisy_arr, window_size=args.window, noise_variance=args.noise_var
    )
    print(f"    Rango de salida: [{arr_out.min():.2f}, {arr_out.max():.2f}]")

    # Paso 4 — Guardar NIfTI filtrado
    print("--- 4/5 Guardando NIfTI ---")
    itk_output = numpy_to_itk(arr_out, reference=itk_input)
    save_image(itk_output, str(out_path))

    # Paso 5 — Guardar PNG de comparación
    print("--- 5/5 Guardando PNG de comparación ---")
    png_path = OUTPUT_COMPARISON_DIR / f"{_get_stem(input_path)}_wiener_comparison.png"
    save_comparison_png(
        arr_in,
        noisy_arr,
        arr_out,
        title=f"{_get_stem(input_path)} — ruido={args.noise_type}  ventana={args.window}",
        out_path=png_path,
        noise_type=args.noise_type,
    )

    # PSNR opcional — solo si se proporciona imagen de referencia
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
            print(f"    AVISO: shape ref {ref_arr.shape} != salida {arr_out.shape}; PSNR omitido")

    print("\n  Pipeline completo.")

    # Limpiar directorio temporal usado para sortear rutas no-ASCII de ITK
    if _TMP_DIR is not None and _TMP_DIR.exists():
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
