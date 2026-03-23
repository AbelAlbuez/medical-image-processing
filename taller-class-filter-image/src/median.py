#!/usr/bin/env python
"""
Filtro de Mediana 3D — implementación propia con NumPy.

Algoritmo (Ali 2018, sección 2.2.1):
    Para cada vóxel, se toma la vecindad cúbica (2*radio+1)³, se excluye el vóxel
    central y se reemplaza el vóxel con la mediana de los vecinos restantes.

    vecindad           = volumen[z-r:z+r+1, y-r:y+r+1, x-r:x+r+1]
    vecinos            = vecindad.flatten()
    vecinos_sin_centro = np.delete(vecinos, indice_centro)
    salida[z, y, x]   = np.median(vecinos_sin_centro)

    La implementación vectoriza este bucle con sliding_window_view para evitar
    iterar en Python vóxel a vóxel (resultado equivalente, mucho más rápido).

Si no se indica salida, guarda en:
    output/median/<base>_median_r{radio}.nii
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import itk
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Directorio temporal para rutas no-ASCII (limitación C++ de ITK en Windows)
_TMP_DIR = None


def _itk_safe(p: Path) -> str:
    """
    Copia el archivo a un directorio temporal con ruta ASCII pura si la ruta
    original contiene caracteres no-ASCII (tildes, ñ, etc.).
    La capa C++ de ITK en Windows no puede abrir rutas con esos caracteres.
    Devuelve la ruta segura como cadena.
    """
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s  # la ruta ya es ASCII, no se necesita copia
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_med_"))
        dst = _TMP_DIR / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        return str(dst)


def _itk_safe_write(p: Path) -> str:
    """
    Devuelve una ruta de escritura ASCII-segura para ITK.
    Si el destino real tiene caracteres no-ASCII, escribe en el directorio
    temporal; el llamador debe mover el archivo al destino final después.
    """
    global _TMP_DIR
    s = str(p)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        if _TMP_DIR is None:
            _TMP_DIR = Path(tempfile.mkdtemp(prefix="itk_med_"))
        return str(_TMP_DIR / p.name)


# Raíz del proyecto: directorio padre de src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_MEDIAN_DIR = PROJECT_ROOT / "output" / "median"


def _basename_from_path(p: Path) -> str:
    """Extrae el nombre base del archivo sin la extensión NIfTI (.nii o .nii.gz)."""
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def resolve_input_path(user_path: str) -> Path:
    """
    Resuelve la ruta de entrada. Si no existe directamente, busca dentro de Images/.
    Acepta rutas absolutas, relativas o solo el nombre del archivo.
    """
    p = Path(user_path)
    if p.is_file():
        return p.resolve()
    alt = PROJECT_ROOT / "Images" / user_path
    if alt.is_file():
        return alt.resolve()
    print(f"ERROR: no se encontró la imagen: {user_path}", file=sys.stderr)
    sys.exit(1)


def resolve_output_path(input_path: Path, output_arg: str | None, radius: int) -> Path:
    """
    Determina la ruta de salida del volumen filtrado.
    Si no se indica salida explícita, usa output/median/<base>_median_r{radio}.nii.
    """
    base = _basename_from_path(input_path)
    if output_arg is None:
        OUTPUT_MEDIAN_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_MEDIAN_DIR / f"{base}_median_r{radius}.nii"
    outp = Path(output_arg)
    if outp.is_absolute():
        outp.parent.mkdir(parents=True, exist_ok=True)
        return outp
    OUTPUT_MEDIAN_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_MEDIAN_DIR / f"{base}_median_r{radius}.nii"


def median_filter_excl_center(arr: np.ndarray, radius: int) -> np.ndarray:
    """
    Filtro de mediana 3D que excluye el vóxel central de cada vecindad.

    Equivalente al bucle explícito de la referencia del taller:

        for z in range(1, dim_z - 1):
            for y in range(1, dim_y - 1):
                for x in range(1, dim_x - 1):
                    vecindad           = volumen[z-1:z+2, y-1:y+2, x-1:x+2]
                    vecinos            = vecindad.flatten()
                    vecinos_sin_centro = np.delete(vecinos, 13)
                    salida[z, y, x]   = np.median(vecinos_sin_centro)

    La implementación vectoriza este bucle con sliding_window_view (sin iterar
    en Python vóxel a vóxel), lo que produce el mismo resultado con mucho menor
    tiempo de cómputo.

    Para radius=1: ventana 3×3×3 (27 vóxeles), el centro está en el índice 13.

    Parámetros
    ----------
    arr    : np.ndarray — volumen 3D de entrada (Z, Y, X), tipo flotante
    radius : int        — radio de la ventana cúbica (radius=1 → ventana 3×3×3)

    Retorna
    -------
    np.ndarray — volumen filtrado, dtype float32, misma forma que arr
    """
    w = 2 * radius + 1
    # Índice del vóxel central en la ventana aplanada (ej: 13 para radius=1, ventana 3³=27)
    center = w ** 3 // 2

    arr_f = arr.astype(np.float32)
    # Relleno reflectivo en los bordes para no perder vóxeles del contorno del volumen
    padded = np.pad(arr_f, radius, mode="reflect")

    # Ventanas deslizantes sobre el volumen con relleno: forma (Z, Y, X, w, w, w)
    windows = sliding_window_view(padded, window_shape=(w, w, w))

    # Aplanar cada ventana cúbica: forma (Z, Y, X, w³)
    flat = windows.reshape(*arr.shape, w ** 3)

    # Eliminar el vóxel central de cada ventana: forma (Z, Y, X, w³ - 1)
    flat_no_center = np.delete(flat, center, axis=-1)

    # Calcular la mediana sobre los vecinos (último eje) y devolver como float32
    return np.median(flat_no_center, axis=-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filtro de mediana 3D propio (NumPy, excluye vóxel central). "
            "Uso: entrada radio [salida]"
        )
    )
    parser.add_argument("input_image", help="Ruta al volumen de entrada (.nii o .nii.gz)")
    parser.add_argument("radius", type=int, help="Radio de la ventana cúbica (ej: 1 → ventana 3×3×3)")
    parser.add_argument(
        "output_image",
        nargs="?",
        default=None,
        help="Ruta de salida opcional (.nii); por defecto output/median/<base>_median_r{radio}.nii",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_image)
    out_path = resolve_output_path(input_path, args.output_image, args.radius)

    # Cargar el volumen como float32 (itk.F) para conservar la precisión de las intensidades MRI
    image = itk.imread(_itk_safe(input_path), itk.F)
    arr = itk.array_view_from_image(image).astype(np.float32)
    print(f"  Cargado : {input_path.name}  shape={arr.shape}")

    # Aplicar el filtro de mediana excluyendo el vóxel central de cada vecindad
    filtered = median_filter_excl_center(arr, args.radius)
    print(f"  Filtrado: radio={args.radius}, ventana={(2*args.radius+1)}³, centro excluido")

    # Reconstruir la imagen ITK copiando los metadatos espaciales del original
    # (espaciado, origen y dirección) para que el NIfTI de salida sea coherente
    out_img = itk.image_view_from_array(filtered)
    out_img.CopyInformation(image)

    # Guardar el volumen filtrado (con manejo de rutas no-ASCII de ITK)
    safe_out = _itk_safe_write(out_path)
    itk.imwrite(out_img, safe_out)

    if safe_out != str(out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(safe_out, out_path)

    print(f"  Guardado: {out_path.name}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Limpiar el directorio temporal creado para sortear rutas no-ASCII
        if _TMP_DIR is not None and Path(_TMP_DIR).exists():
            shutil.rmtree(_TMP_DIR, ignore_errors=True)
