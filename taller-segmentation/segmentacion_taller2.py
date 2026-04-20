"""
segmentacion_taller2.py
Taller 2 — Segmentación por Umbrales
Procesamiento de Imágenes Médicas — Pontificia Universidad Javeriana

Script único y autónomo que ejecuta los tres métodos de segmentación del taller
sobre cada imagen MR disponible en images/:

  1. BinaryThresholdImageFilter       — umbrales manuales por órgano
  2. OtsuMultipleThresholdsImageFilter — n = 1, 2, 3 umbrales automáticos
  3. ScalarImageKmeansImageFilter     — n = 2, 3, 4 clases

Se ejecuta sin argumentos desde la raíz del proyecto taller-segmentation/:

    python segmentacion_taller2.py

Descubre imágenes automáticamente (soporta archivos planos en images/ o
subcarpetas por órgano images/<key>/) y guarda los resultados en
results/<metodo>/<nombre>.nii.gz.

Referencias ITK:
  Binary: https://examples.itk.org/src/filtering/thresholding/thresholdanimage/documentation
  Otsu:   https://examples.itk.org/src/filtering/thresholding/thresholdanimageusingotsu/documentation
  K-Means: https://examples.itk.org/src/segmentation/classifiers/clusterpixelsingrayscaleimage/documentation
"""
from __future__ import annotations

import itk
import numpy as np
from pathlib import Path
from typing import Optional

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
IMAGES_DIR  = ROOT / "images"
RESULTS_DIR = ROOT / "results"

# Claves de cada imagen (según consigna del taller)
IMAGE_KEYS = ["brain_tumor", "breast_cancer", "liver_tumor"]

# Patrones de búsqueda en el nombre del archivo (insensible a mayúsculas).
# Permite descubrir imágenes cuando están planas en images/ (ej. MRBrainTumor.nii.gz)
# o dentro de una subcarpeta por órgano (images/brain_tumor/*.nii).
KEY_PATTERNS = {
    "brain_tumor":   ["brain"],
    "breast_cancer": ["breast"],
    "liver_tumor":   ["liver"],
}

# Binary Threshold — valores manuales obtenidos con 3D Slicer (taller)
BINARY_PARAMS = {
    "brain_tumor":   {"lower": 173, "upper": 360, "inside": 1, "outside": 0},
    "breast_cancer": {"lower": 330, "upper": 750, "inside": 1, "outside": 0},
    "liver_tumor":   {"lower":  44, "upper":  85, "inside": 1, "outside": 0},
}

# Otsu — número de umbrales a probar por imagen
OTSU_N_THRESHOLDS   = [1, 2, 3]
OTSU_HISTOGRAM_BINS = 128

# K-Means — número de clases a probar por imagen (máximo 4)
KMEANS_N_CLASSES = [2, 3, 4]


# ── DESCUBRIMIENTO DE IMÁGENES ────────────────────────────────────────────────

def _find_image_for_key(key: str, images_dir: Path) -> Optional[Path]:
    """Busca el NIfTI/MHA asociado a una clave.

    Orden de búsqueda:
      1. Subcarpeta images/<key>/ (primer .nii, .nii.gz o .mha encontrado)
      2. Archivo plano en images/ cuyo nombre contenga el patrón asociado.
    """
    subdir = images_dir / key
    if subdir.is_dir():
        matches = sorted(
            list(subdir.glob("*.nii"))
            + list(subdir.glob("*.nii.gz"))
            + list(subdir.glob("*.mha"))
        )
        if matches:
            return matches[0]

    patrones = KEY_PATTERNS.get(key, [key])
    candidatos = sorted(
        [p for p in images_dir.iterdir()
         if p.is_file() and (p.suffix == ".mha" or ".nii" in p.name.lower())]
    )
    for p in candidatos:
        nombre = p.name.lower()
        if any(pat in nombre for pat in patrones):
            return p
    return None


# ── 1. BINARY THRESHOLD ───────────────────────────────────────────────────────

def run_binary_threshold(image_path: Path, params: dict, output_dir: Path) -> Path:
    """Aplica BinaryThresholdImageFilter con los umbrales dados.

    Todos los vóxeles en [lower, upper] toman `inside`; el resto toma `outside`.
    Guarda el resultado en output_dir/<stem>_binary.nii.gz.
    """
    lower   = int(params["lower"])
    upper   = int(params["upper"])
    inside  = int(params["inside"])
    outside = int(params["outside"])

    print(f"  [Binary] lower={lower}  upper={upper}  "
          f"inside={inside}  outside={outside}")

    InputImageType  = itk.Image[itk.F, 3]
    OutputImageType = itk.Image[itk.UC, 3]

    reader = itk.ImageFileReader[InputImageType].New()
    reader.SetFileName(str(image_path))

    filtro = itk.BinaryThresholdImageFilter[InputImageType, OutputImageType].New()
    filtro.SetInput(reader.GetOutput())
    filtro.SetLowerThreshold(lower)
    filtro.SetUpperThreshold(upper)
    filtro.SetInsideValue(inside)
    filtro.SetOutsideValue(outside)
    filtro.Update()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem_sin_ext(image_path)
    ruta_out = output_dir / f"{stem}_binary.nii.gz"
    itk.imwrite(filtro.GetOutput(), str(ruta_out))
    print(f"  [Binary] → {ruta_out.relative_to(ROOT)}")
    return ruta_out


# ── 2. OTSU MULTI-UMBRAL ──────────────────────────────────────────────────────

def run_otsu(image_path: Path, n_thresholds: int, output_dir: Path) -> Path:
    """Aplica OtsuMultipleThresholdsImageFilter con n umbrales automáticos.

    Genera una segmentación con n+1 regiones (etiquetas 0..n) e imprime
    los umbrales calculados por ITK.
    Guarda en output_dir/<stem>_otsu_<n>.nii.gz.
    """
    imagen = itk.imread(str(image_path), itk.F)
    IT     = type(imagen)

    filtro = itk.OtsuMultipleThresholdsImageFilter[IT, IT].New()
    filtro.SetInput(imagen)
    filtro.SetNumberOfThresholds(n_thresholds)
    filtro.SetNumberOfHistogramBins(OTSU_HISTOGRAM_BINS)
    filtro.Update()

    umbrales = [round(float(u), 2) for u in filtro.GetThresholds()]
    resultado = filtro.GetOutput()
    arr_seg   = itk.array_from_image(resultado)

    print(f"  [Otsu n={n_thresholds}] umbrales={umbrales}  "
          f"regiones={n_thresholds + 1}")
    for et in range(n_thresholds + 1):
        cnt = int(np.sum(arr_seg == et))
        pct = cnt / arr_seg.size * 100
        print(f"          Et.{et}: {cnt:>9,} vóx ({pct:5.2f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem_sin_ext(image_path)
    ruta_out = output_dir / f"{stem}_otsu_{n_thresholds}.nii.gz"
    itk.imwrite(resultado, str(ruta_out))
    print(f"  [Otsu n={n_thresholds}] → {ruta_out.relative_to(ROOT)}")
    return ruta_out


# ── 3. K-MEANS ────────────────────────────────────────────────────────────────

def run_kmeans(image_path: Path, n_classes: int, output_dir: Path) -> Path:
    """Aplica ScalarImageKmeansImageFilter con n_classes clusters.

    Centroides iniciales: `n_classes` valores distribuidos uniformemente entre
    el mínimo y el máximo de las intensidades no-cero. Imprime centroides
    iniciales y finales.
    Guarda en output_dir/<stem>_kmeans_<n>.nii.gz.
    """
    if n_classes > 4:
        raise ValueError(f"K-Means: máximo 4 clases permitidas, recibido {n_classes}")

    imagen = itk.imread(str(image_path), itk.F)
    arr    = itk.array_from_image(imagen)
    validos = arr[arr > 0]
    centroides_iniciales = np.linspace(
        float(validos.min()), float(validos.max()), n_classes
    ).tolist()

    InputImageType  = itk.Image[itk.F, 3]
    OutputImageType = itk.Image[itk.UC, 3]

    filtro = itk.ScalarImageKmeansImageFilter[InputImageType, OutputImageType].New()
    filtro.SetInput(imagen)
    filtro.SetUseNonContiguousLabels(True)
    for c in centroides_iniciales:
        filtro.AddClassWithInitialMean(c)
    filtro.Update()

    centroides_finales = [round(float(m), 2) for m in filtro.GetFinalMeans()]
    ini_fmt = [round(c, 2) for c in centroides_iniciales]

    print(f"  [KMeans k={n_classes}] centroides ini={ini_fmt}")
    print(f"  [KMeans k={n_classes}] centroides fin={centroides_finales}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem_sin_ext(image_path)
    ruta_out = output_dir / f"{stem}_kmeans_{n_classes}.nii.gz"
    itk.imwrite(filtro.GetOutput(), str(ruta_out))
    print(f"  [KMeans k={n_classes}] → {ruta_out.relative_to(ROOT)}")
    return ruta_out


# ── UTILIDADES ────────────────────────────────────────────────────────────────

def _stem_sin_ext(path: Path) -> str:
    """Devuelve el nombre del archivo sin ninguna extensión (.nii, .nii.gz, .mha)."""
    nombre = path.name
    for ext in (".nii.gz", ".nii", ".mha"):
        if nombre.lower().endswith(ext):
            return nombre[: -len(ext)]
    return path.stem


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  TALLER 2 — SEGMENTACIÓN POR UMBRALES")
    print("  Binary Threshold · Otsu · K-Means")
    print("=" * 70)
    print(f"  IMAGES_DIR : {IMAGES_DIR}")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")

    if not IMAGES_DIR.is_dir():
        raise SystemExit(f"  ✗ No existe el directorio de imágenes: {IMAGES_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dir_binary = RESULTS_DIR / "binary_threshold"
    dir_otsu   = RESULTS_DIR / "otsu"
    dir_kmeans = RESULTS_DIR / "kmeans"

    generados: list[Path] = []

    for key in IMAGE_KEYS:
        img_path = _find_image_for_key(key, IMAGES_DIR)
        print()
        print("─" * 70)
        print(f"  {key.upper()}")
        print("─" * 70)

        if img_path is None:
            print(f"  ⚠  No se encontró imagen para '{key}' en {IMAGES_DIR}")
            continue

        print(f"  Imagen: {img_path.relative_to(ROOT)}")

        # 1. Binary Threshold
        try:
            out = run_binary_threshold(img_path, BINARY_PARAMS[key], dir_binary)
            generados.append(out)
        except Exception as e:
            print(f"  ✗ Binary falló: {e}")

        # 2. Otsu (varios n)
        for n in OTSU_N_THRESHOLDS:
            try:
                out = run_otsu(img_path, n, dir_otsu)
                generados.append(out)
            except Exception as e:
                print(f"  ✗ Otsu n={n} falló: {e}")

        # 3. K-Means (varios k)
        for k in KMEANS_N_CLASSES:
            try:
                out = run_kmeans(img_path, k, dir_kmeans)
                generados.append(out)
            except Exception as e:
                print(f"  ✗ K-Means k={k} falló: {e}")

    # Resumen
    print()
    print("=" * 70)
    print(f"  RESUMEN — {len(generados)} archivos generados")
    print("=" * 70)
    for p in generados:
        print(f"    {p.relative_to(ROOT)}")
    print("=" * 70)
