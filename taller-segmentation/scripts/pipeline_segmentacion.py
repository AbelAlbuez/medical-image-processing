"""
pipeline_segmentacion.py
Taller 2 - Segmentación por Umbrales
Procesamiento de Imágenes Médicas - Pontificia Universidad Javeriana

Pipeline completo de segmentación:
  1. Analiza histograma de cada imagen y deriva rangos fundamentados
  2. Aplica BinaryThreshold, Otsu y K-means
  3. Genera múltiples vistas (axial, coronal, sagital) por cada configuración

Los rangos del umbral binario NO son arbitrarios: se derivan automáticamente
a partir de los percentiles del histograma de cada imagen.

Uso: python scripts/pipeline_segmentacion.py
"""

import os
import itk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ─── Configuración ────────────────────────────────────────────────────────────
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")

IMAGES = {
    "brain":  os.path.join(IMAGES_DIR, "MRBrainTumor.nii.gz"),
    "breast": os.path.join(IMAGES_DIR, "MRBreastCancer.nii.gz"),
    "liver":  os.path.join(IMAGES_DIR, "MRLiverTumor.nii.gz"),
}

LABELS = {
    "brain":  "MR Tumor Cerebral",
    "breast": "MR Cáncer de Mama",
    "liver":  "MR Tumor Hepático",
}

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(RAIZ, "report", "figures")
RESULTS_DIR = os.path.join(RAIZ, "results")

# ──────────────────────────────────────────────────────────────────────────────
# Estrategia de rangos por imagen (justificada por histograma):
#
# CEREBRO (T1c con contraste):
#   - Tejido normal: pico principal en intensidades medias
#   - Tumor realzado: hiperintenso → cola derecha del histograma
#   - Rango: P75–P99 (captura solo los picos más brillantes = realce de contraste)
#   - Rango_estrecho: P85–P99 (más selectivo, reduce falsos positivos)
#
# MAMA:
#   - Tejido graso: pico dominante en intensidades bajas/medias
#   - Tumor: masa con intensidad diferente al tejido circundante
#   - Rango: P70–P95 (región por encima del tejido normal)
#   - Rango_estrecho: P80–P95 (más selectivo)
#
# HÍGADO:
#   - Parénquima hepático: homogéneo, pico principal bien definido
#   - Tumor hepático: frecuentemente hipointenso (más oscuro que el hígado)
#   - Rango: P30–P65 (por debajo del pico principal del hígado)
#   - Rango_estrecho: P40–P60 (zona más específica del tumor)
# ──────────────────────────────────────────────────────────────────────────────

# Percentiles usados para derivar rangos de umbral por imagen
ESTRATEGIA_UMBRALES = {
    "brain":  {"lower_pct": 75, "upper_pct": 99, "lower_pct2": 85, "upper_pct2": 99},
    "breast": {"lower_pct": 70, "upper_pct": 95, "lower_pct2": 80, "upper_pct2": 95},
    "liver":  {"lower_pct": 30, "upper_pct": 65, "lower_pct2": 40, "upper_pct2": 60},
}

# K-means: máximo 4 clases
NUM_CLASSES_LIST   = [2, 3, 4]
# Otsu: 1, 2, 3 umbrales
NUM_THRESHOLDS_LIST = [1, 2, 3]
NUM_HISTOGRAM_BINS  = 128

# ─── Utilidades ───────────────────────────────────────────────────────────────

def crear_directorios():
    for carpeta in [
        FIGURES_DIR,
        os.path.join(RESULTS_DIR, "binary_threshold"),
        os.path.join(RESULTS_DIR, "otsu"),
        os.path.join(RESULTS_DIR, "kmeans"),
    ]:
        os.makedirs(carpeta, exist_ok=True)


def cargar_imagen(path):
    """Carga imagen como float (pixel type itk.F obligatorio)."""
    return itk.imread(path, itk.F)


def calcular_estadisticas(arr):
    """Calcula estadísticas sobre los vóxeles válidos (excluye fondo=0)."""
    validos = arr[arr > 0].flatten()
    pcts = np.percentile(validos, [25, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 99])
    return {
        "validos": validos,
        "min": float(validos.min()),
        "max": float(validos.max()),
        "media": float(validos.mean()),
        "std": float(validos.std()),
        "p25": pcts[0],  "p30": pcts[1],  "p40": pcts[2],
        "p50": pcts[3],  "p60": pcts[4],  "p65": pcts[5],
        "p70": pcts[6],  "p75": pcts[7],  "p80": pcts[8],
        "p85": pcts[9],  "p90": pcts[10], "p95": pcts[11],
        "p99": pcts[12],
    }


def derivar_rangos(key, stats):
    """Deriva rangos de umbral binario a partir de los percentiles del histograma."""
    e = ESTRATEGIA_UMBRALES[key]
    lower1 = float(stats[f"p{e['lower_pct']}"])
    upper1 = float(stats[f"p{e['upper_pct']}"])
    lower2 = float(stats[f"p{e['lower_pct2']}"])
    upper2 = float(stats[f"p{e['upper_pct2']}"])
    return [
        {"lower": lower1, "upper": upper1,
         "etiqueta": f"P{e['lower_pct']}–P{e['upper_pct']} (amplio)"},
        {"lower": lower2, "upper": upper2,
         "etiqueta": f"P{e['lower_pct2']}–P{e['upper_pct2']} (estrecho)"},
    ]


def encontrar_slice_tumor(arr, key):
    """
    Heurística para encontrar el slice axial con mayor actividad (probable tumor).
    Para cada eje devuelve el slice con mayor varianza de intensidad.
    """
    # Slice axial: mayor varianza a lo largo de Z
    varianzas_z = [np.var(arr[z, :, :]) for z in range(arr.shape[0])]
    slice_z = int(np.argmax(varianzas_z))

    # Slice coronal: mayor varianza a lo largo de Y
    varianzas_y = [np.var(arr[:, y, :]) for y in range(arr.shape[1])]
    slice_y = int(np.argmax(varianzas_y))

    # Slice sagital: mayor varianza a lo largo de X
    varianzas_x = [np.var(arr[:, :, x]) for x in range(arr.shape[2])]
    slice_x = int(np.argmax(varianzas_x))

    return slice_z, slice_y, slice_x


def generar_figura_multivista(arr_original, resultados, key, nombre_figura, subtitulos):
    """
    Genera figura con múltiples vistas ortogonales para comparar métodos.

    Columnas: axial | coronal | sagital
    Filas: imagen original + cada resultado de segmentación
    """
    # Encontrar slices con mayor probabilidad de contener el tumor
    sz, sy, sx = encontrar_slice_tumor(arr_original, key)

    n_filas = 1 + len(resultados)
    fig = plt.figure(figsize=(15, 4 * n_filas))
    fig.suptitle(
        f"{LABELS[key]} — Comparación de métodos\n"
        f"Slices automáticos: Axial Z={sz}, Coronal Y={sy}, Sagital X={sx}",
        fontsize=13, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(n_filas, 3, figure=fig, hspace=0.45, wspace=0.3)

    ejes_nombres = ["Axial (Z)", "Coronal (Y)", "Sagital (X)"]

    def agregar_fila(fila_idx, arr, titulo, cmap):
        vistas = [
            arr[sz, :, :],
            arr[:, sy, :],
            arr[:, :, sx],
        ]
        for col, (vista, eje_nombre) in enumerate(zip(vistas, ejes_nombres)):
            ax = fig.add_subplot(gs[fila_idx, col])
            im = ax.imshow(vista, cmap=cmap, origin="lower", aspect="equal")
            if fila_idx == 0:
                ax.set_title(eje_nombre, fontsize=10, fontweight="bold")
            ax.set_ylabel(titulo if col == 0 else "", fontsize=8, rotation=90, labelpad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Fila 0: imagen original
    agregar_fila(0, arr_original, "Original", "gray")

    # Filas siguientes: resultados de segmentación
    for i, (arr_seg, subtitulo) in enumerate(zip(resultados, subtitulos)):
        agregar_fila(i + 1, arr_seg, subtitulo, "tab10")

    fig.tight_layout()
    ruta = os.path.join(FIGURES_DIR, f"{key}_{nombre_figura}.png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    → Figura guardada: {os.path.basename(ruta)}")
    return ruta


# ─── Paso 1: Histograma y estadísticas ────────────────────────────────────────

def paso_histograma(key, path):
    print(f"\n  [HISTOGRAMA] {LABELS[key]}")
    imagen = cargar_imagen(path)
    arr = itk.array_from_image(imagen)
    stats = calcular_estadisticas(arr)

    print(f"    Min={stats['min']:.1f}  Max={stats['max']:.1f}  "
          f"Media={stats['media']:.1f}  Std={stats['std']:.1f}")
    print(f"    P30={stats['p30']:.1f}  P50={stats['p50']:.1f}  "
          f"P75={stats['p75']:.1f}  P85={stats['p85']:.1f}  "
          f"P95={stats['p95']:.1f}  P99={stats['p99']:.1f}")

    rangos = derivar_rangos(key, stats)
    print(f"    Rangos derivados:")
    for r in rangos:
        print(f"      {r['etiqueta']}: [{r['lower']:.1f}, {r['upper']:.1f}]")

    # Graficar histograma
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(stats["validos"], bins=100, color="#4C72B0", alpha=0.8,
            edgecolor="white", linewidth=0.3)

    colores = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
    for r, c in zip(rangos, colores[:2]):
        ax.axvspan(r["lower"], r["upper"], alpha=0.2, color=c, label=r["etiqueta"])
        ax.axvline(r["lower"], color=c, linestyle="--", linewidth=1.2)
        ax.axvline(r["upper"], color=c, linestyle="--", linewidth=1.2)

    for pct_key, color, label in [
        ("p50", "#f39c12", "P50"), ("p75", "#e74c3c", "P75"),
        ("p95", "#8e44ad", "P95"), ("p99", "#2c3e50", "P99"),
    ]:
        ax.axvline(stats[pct_key], color=color, linestyle=":", linewidth=1,
                   label=f"{label}={stats[pct_key]:.1f}")

    ax.set_title(f"Histograma — {LABELS[key]}\n"
                 f"Zonas sombreadas = rangos de umbral binario derivados",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Intensidad (vóxel float)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"histogram_{key}.png"), dpi=150)
    plt.close(fig)

    return imagen, arr, stats, rangos


# ─── Paso 2: Umbral Binario ───────────────────────────────────────────────────

def paso_binary_threshold(key, imagen, arr, rangos):
    print(f"\n  [BINARY THRESHOLD] {LABELS[key]}")
    ImageType = type(imagen)

    resultados_arr = []
    subtitulos     = []

    for r in rangos:
        print(f"    Rango {r['etiqueta']}: [{r['lower']:.1f}, {r['upper']:.1f}]")

        filtro = itk.BinaryThresholdImageFilter[ImageType, ImageType].New()
        filtro.SetInput(imagen)
        filtro.SetLowerThreshold(r["lower"])
        filtro.SetUpperThreshold(r["upper"])
        filtro.SetInsideValue(1)
        filtro.SetOutsideValue(0)
        filtro.Update()

        arr_res = itk.array_from_image(filtro.GetOutput())
        n_seg = int(np.sum(arr_res == 1))
        print(f"      Vóxeles segmentados: {n_seg:,} ({100*n_seg/arr_res.size:.2f}%)")

        nombre_archivo = f"{key}_binary_{r['lower']:.0f}_{r['upper']:.0f}.nii.gz"
        itk.imwrite(filtro.GetOutput(),
                    os.path.join(RESULTS_DIR, "binary_threshold", nombre_archivo))

        resultados_arr.append(arr_res)
        subtitulos.append(f"Binary\n{r['etiqueta']}")

    generar_figura_multivista(arr, resultados_arr, key, "binary_comparacion", subtitulos)


# ─── Paso 3: Otsu ─────────────────────────────────────────────────────────────

def paso_otsu(key, imagen, arr):
    print(f"\n  [OTSU] {LABELS[key]}")
    ImageType = type(imagen)

    resultados_arr = []
    subtitulos     = []

    for n in NUM_THRESHOLDS_LIST:
        filtro = itk.OtsuMultipleThresholdsImageFilter[ImageType, ImageType].New()
        filtro.SetInput(imagen)
        filtro.SetNumberOfThresholds(n)
        filtro.SetNumberOfHistogramBins(NUM_HISTOGRAM_BINS)
        filtro.Update()

        umbrales = [round(float(u), 2) for u in filtro.GetThresholds()]
        print(f"    n={n} umbral(es): {umbrales}")

        arr_res = itk.array_from_image(filtro.GetOutput())
        for et in range(n + 1):
            print(f"      Etiqueta {et}: {int(np.sum(arr_res == et)):,} vóxeles")

        itk.imwrite(filtro.GetOutput(),
                    os.path.join(RESULTS_DIR, "otsu", f"{key}_otsu_{n}.nii.gz"))

        resultados_arr.append(arr_res)
        subtitulos.append(f"Otsu n={n}\nUmbrales: {umbrales}")

    generar_figura_multivista(arr, resultados_arr, key, "otsu_comparacion", subtitulos)


# ─── Paso 4: K-means ──────────────────────────────────────────────────────────

def paso_kmeans(key, imagen, arr):
    print(f"\n  [K-MEANS] {LABELS[key]}")
    ImageType = type(imagen)

    validos  = arr[arr > 0].flatten()
    min_val  = float(validos.min())
    max_val  = float(validos.max())

    resultados_arr = []
    subtitulos     = []

    for n in NUM_CLASSES_LIST:
        # Centroides iniciales distribuidos uniformemente en el rango válido
        centroides = np.linspace(min_val, max_val, n).tolist()

        filtro = itk.ScalarImageKmeansImageFilter[itk.Image[itk.F, 3], itk.Image[itk.UC, 3]].New()
        filtro.SetInput(imagen)
        for c in centroides:
            filtro.AddClassWithInitialMean(c)
        filtro.Update()

        finales = [round(float(m), 1) for m in filtro.GetFinalMeans()]
        print(f"    n={n} clases → centroides finales: {finales}")

        arr_res = itk.array_from_image(filtro.GetOutput())
        for et in range(n):
            print(f"      Etiqueta {et} (centroide={finales[et] if et < len(finales) else '?'}): "
                  f"{int(np.sum(arr_res == et)):,} vóxeles")

        itk.imwrite(filtro.GetOutput(),
                    os.path.join(RESULTS_DIR, "kmeans", f"{key}_kmeans_{n}.nii.gz"))

        resultados_arr.append(arr_res)
        subtitulos.append(f"K-means n={n}\nCentroides: {finales}")

    generar_figura_multivista(arr, resultados_arr, key, "kmeans_comparacion", subtitulos)


# ─── Paso 5: Figura resumen comparativa (los 3 métodos juntos) ────────────────

def paso_resumen_comparativo(key, imagen, arr, rangos):
    """
    Genera una figura final comparando el mejor resultado de cada método
    en las 3 vistas ortogonales.
    """
    print(f"\n  [RESUMEN COMPARATIVO] {LABELS[key]}")
    ImageType = type(imagen)

    # Mejor umbral binario: el rango estrecho (índice 1)
    r = rangos[1]
    filtro_bin = itk.BinaryThresholdImageFilter[ImageType, ImageType].New()
    filtro_bin.SetInput(imagen)
    filtro_bin.SetLowerThreshold(r["lower"])
    filtro_bin.SetUpperThreshold(r["upper"])
    filtro_bin.SetInsideValue(1)
    filtro_bin.SetOutsideValue(0)
    filtro_bin.Update()
    arr_bin = itk.array_from_image(filtro_bin.GetOutput())

    # Mejor Otsu: 2 umbrales (3 regiones)
    filtro_otsu = itk.OtsuMultipleThresholdsImageFilter[ImageType, ImageType].New()
    filtro_otsu.SetInput(imagen)
    filtro_otsu.SetNumberOfThresholds(2)
    filtro_otsu.SetNumberOfHistogramBins(NUM_HISTOGRAM_BINS)
    filtro_otsu.Update()
    arr_otsu = itk.array_from_image(filtro_otsu.GetOutput())

    # Mejor K-means: 3 clases
    validos = arr[arr > 0].flatten()
    centroides = np.linspace(float(validos.min()), float(validos.max()), 3).tolist()
    filtro_km = itk.ScalarImageKmeansImageFilter[itk.Image[itk.F, 3], itk.Image[itk.UC, 3]].New()
    filtro_km.SetInput(imagen)
    for c in centroides:
        filtro_km.AddClassWithInitialMean(c)
    filtro_km.Update()
    arr_km = itk.array_from_image(filtro_km.GetOutput())

    resultados = [arr_bin, arr_otsu, arr_km]
    subtitulos = [
        f"Binary [{r['lower']:.0f}–{r['upper']:.0f}]",
        "Otsu (n=2 umbrales)",
        "K-means (n=3 clases)",
    ]
    generar_figura_multivista(arr, resultados, key, "resumen_comparativo", subtitulos)


# ─── Ejecución principal ──────────────────────────────────────────────────────

def main():
    inicio = datetime.now()
    print("\n" + "="*60)
    print("  PIPELINE DE SEGMENTACIÓN POR UMBRALES — TALLER 2")
    print("  Procesamiento de Imágenes Médicas — Javeriana")
    print(f"  Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    crear_directorios()

    for key, path in IMAGES.items():
        print(f"\n{'='*60}")
        print(f"  IMAGEN: {LABELS[key]}")
        print(f"{'='*60}")

        if not os.path.exists(path):
            print(f"  ⚠ Archivo no encontrado: {path}")
            print(f"  Saltando esta imagen...")
            continue

        imagen, arr, stats, rangos = paso_histograma(key, path)
        paso_binary_threshold(key, imagen, arr, rangos)
        paso_otsu(key, imagen, arr)
        paso_kmeans(key, imagen, arr)
        paso_resumen_comparativo(key, imagen, arr, rangos)

    fin = datetime.now()
    duracion = (fin - inicio).seconds

    print("\n" + "="*60)
    print("  ✓ PIPELINE COMPLETADO")
    print(f"  Duración total: {duracion}s")
    print("  Figuras generadas en: report/figures/")
    print("  Por cada imagen se generaron:")
    print("    - histogram_<imagen>.png           (histograma + rangos)")
    print("    - <imagen>_binary_comparacion.png  (2 rangos × 3 vistas)")
    print("    - <imagen>_otsu_comparacion.png    (3 configs × 3 vistas)")
    print("    - <imagen>_kmeans_comparacion.png  (3 configs × 3 vistas)")
    print("    - <imagen>_resumen_comparativo.png (mejor de cada método)")
    print("="*60)


if __name__ == "__main__":
    main()
