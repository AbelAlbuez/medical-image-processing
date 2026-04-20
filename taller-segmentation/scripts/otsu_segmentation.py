"""
otsu_segmentation.py
Taller 2 — Segmentación por Umbrales
Procesamiento de Imágenes Médicas — Pontificia Universidad Javeriana

Aplica OtsuMultipleThresholdsImageFilter con n = 1, 2, 3.
Guarda los resultados en results/otsu/ para revisión manual en 3D Slicer.

Referencia oficial ITK:
  https://examples.itk.org/src/filtering/thresholding/thresholdanimageusingotsu/documentation
"""

import os
import itk
import numpy as np

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")

IMAGES = {
    "brain":  os.path.join(IMAGES_DIR, "MRBrainTumor.nii.gz"),
    "breast": os.path.join(IMAGES_DIR, "MRBreastCancer.nii.gz"),
    "liver":  os.path.join(IMAGES_DIR, "MRLiverTumor.nii.gz"),
}

LABELS_ES = {
    "brain":  "Tumor cerebral (MR)",
    "breast": "Cáncer de mama (MR)",
    "liver":  "Tumor hepático (MR)",
}

NUM_THRESHOLDS_LIST = [1, 2, 3]
NUM_HISTOGRAM_BINS  = 128

RAIZ        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(RAIZ, "..", "results", "otsu")
os.makedirs(RESULTS_DIR, exist_ok=True)


def aplicar_otsu(key, path):
    print(f"\n{'='*60}")
    print(f"  {LABELS_ES[key].upper()}")
    print(f"{'='*60}")

    imagen = itk.imread(path, itk.F)
    arr    = itk.array_from_image(imagen)
    IT     = type(imagen)

    print(f"  Shape : {arr.shape}")
    print(f"  Rango : min={arr.min():.1f}  max={arr.max():.1f}")
    print(f"  P50={np.percentile(arr[arr>0],50):.1f}  "
          f"P75={np.percentile(arr[arr>0],75):.1f}  "
          f"P90={np.percentile(arr[arr>0],90):.1f}  "
          f"P99={np.percentile(arr[arr>0],99):.1f}")

    for n in NUM_THRESHOLDS_LIST:
        filtro = itk.OtsuMultipleThresholdsImageFilter[IT, IT].New()
        filtro.SetInput(imagen)
        filtro.SetNumberOfThresholds(n)
        filtro.SetNumberOfHistogramBins(NUM_HISTOGRAM_BINS)
        filtro.Update()

        umbrales  = [round(float(u), 2) for u in filtro.GetThresholds()]
        resultado = filtro.GetOutput()
        arr_seg   = itk.array_from_image(resultado)

        print(f"\n  n={n}")
        print(f"    Umbrales calculados : {umbrales}")
        print(f"    Regiones generadas  : {n+1} (etiquetas 0 a {n})")
        print(f"    Rangos de intensidad por etiqueta:")

        limites = [0.0] + umbrales + [float(arr.max())]
        for et in range(n + 1):
            cnt   = int(np.sum(arr_seg == et))
            pct   = cnt / arr_seg.size * 100
            rango = f"[{limites[et]:.1f} – {limites[et+1]:.1f}]"
            print(f"      Et.{et}: {cnt:>9,} vóx ({pct:5.2f}%)  intensidad {rango}")

        ruta = os.path.join(RESULTS_DIR, f"{key}_otsu_{n}.nii.gz")
        itk.imwrite(resultado, ruta)
        print(f"    Guardado: {os.path.basename(ruta)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SEGMENTACIÓN — MÉTODO DE OTSU")
    print("  Taller 2 | Procesamiento de Imágenes Médicas")
    print("  Pontificia Universidad Javeriana — 2026")
    print("="*60)

    for key, path in IMAGES.items():
        if not os.path.exists(path):
            print(f"\n  ⚠  No encontrado: {path}")
            continue
        aplicar_otsu(key, path)

    print("\n" + "="*60)
    print("  ✓  Completado — 9 archivos NII en results/otsu/")
    print("="*60)
    print()
    print("  ARCHIVOS GENERADOS:")
    for key in IMAGES:
        for n in NUM_THRESHOLDS_LIST:
            print(f"    results/otsu/{key}_otsu_{n}.nii.gz")
    print()
    print("  PRÓXIMO PASO — EN 3D SLICER:")
    print("    1. File → Add Data → abrir cada NII")
    print("    2. Cambiar visualización a Label Map")
    print("    3. Identificar qué etiqueta (0,1,2,3) es el tumor")
    print("    4. Anotar: imagen, n, etiqueta_tumor")
    print("="*60)
