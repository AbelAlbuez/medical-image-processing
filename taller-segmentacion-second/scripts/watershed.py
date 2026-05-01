import itk
import argparse
import math
import os
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path

# Maximo de celdas por fila en el mosaico final
MAX_CELDAS_POR_FILA = 4

# Segmentacion por Watersheds usando ITK - Barrido de parametros
# Referencia: https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch4.html#x35-1860004.2

# Parametros generales
Dimension = 3
PixelType = itk.F  # float, requerido por GradientMagnitudeRecursiveGaussian
ImageType = itk.Image[PixelType, Dimension]

# ---------------------------------------------------------------------------
# Valores por defecto del barrido
# ---------------------------------------------------------------------------
SIGMA_VALUES     = [0.5, 1.0, 2.0]   # suavidad del gradiente
THRESHOLD_VALUES = [0.005, 0.01, 0.05]  # umbral de gradientes (descarte de ruido)
LEVEL_VALUES     = [0.1, 0.2, 0.4]   # nivel de inundacion (fusion de cuencas)
# ---------------------------------------------------------------------------


def parsear_argumentos():
    # Configuracion CLI con argparse - cada flag acepta multiples valores
    parser = argparse.ArgumentParser(
        description="Segmentacion 3D por Watersheds con ITK - barrido de parametros"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Ruta al archivo de entrada (.nii.gz). Si es relativa, se resuelve contra ./images/",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=SIGMA_VALUES,
        help=f"Uno o mas valores de sigma del gradiente (default: {SIGMA_VALUES})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=THRESHOLD_VALUES,
        help=f"Uno o mas umbrales minimos de gradiente (default: {THRESHOLD_VALUES})",
    )
    parser.add_argument(
        "--level",
        type=float,
        nargs="+",
        default=LEVEL_VALUES,
        help=f"Uno o mas niveles de inundacion (default: {LEVEL_VALUES})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/watershed/",
        help="Directorio de salida (default: output/watershed/)",
    )
    return parser.parse_args()


def resolver_ruta_entrada(ruta_str):
    # Si la ruta no es absoluta, se busca dentro de la carpeta images/
    ruta = Path(ruta_str)
    if not ruta.is_absolute() and not ruta.exists():
        candidato = Path("images") / ruta
        if candidato.exists():
            return candidato
    return ruta


def calcular_gradiente(imagen, sigma):
    # Paso intermedio - magnitud del gradiente gaussiano
    # sigma controla la suavidad del calculo (los bordes son zonas de alto gradiente)
    GradientFilter = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType]
    gradient = GradientFilter.New()
    gradient.SetInput(imagen)
    gradient.SetSigma(sigma)
    gradient.Update()
    return gradient.GetOutput()


def aplicar_watershed(imagen_gradiente, threshold, level):
    # Filtro Watershed sobre el mapa de gradiente
    # threshold descarta gradientes muy bajos (ruido)
    # level controla la fusion de cuencas
    WatershedFilter = itk.WatershedImageFilter[ImageType]
    watershed = WatershedFilter.New()
    watershed.SetInput(imagen_gradiente)
    watershed.SetThreshold(threshold)
    watershed.SetLevel(level)
    watershed.Update()
    return watershed.GetOutput()


def figura_individual(array_etiquetas, ruta_salida, titulo):
    # Genera figura 1x3 con cortes axial, coronal y sagital del label map
    # NOTA: el WatershedImageFilter produce un label map con muchas etiquetas
    # (oversegmentation es normal). El colormap es esencial para interpretar el
    # resultado: sin color, las regiones adyacentes son indistinguibles porque
    # los valores de etiqueta consecutivos lucen casi identicos en escala de grises.
    nz, ny, nx = array_etiquetas.shape
    z_central = nz // 2
    y_central = ny // 2
    x_central = nx // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(np.rot90(array_etiquetas[z_central, :, :]),
                   cmap="nipy_spectral", aspect="auto")
    axes[0].set_title(f"Axial (z={z_central})")
    axes[1].imshow(np.rot90(array_etiquetas[:, y_central, :]),
                   cmap="nipy_spectral", aspect="auto")
    axes[1].set_title(f"Coronal (y={y_central})")
    axes[2].imshow(array_etiquetas[:, :, x_central],
                   cmap="nipy_spectral", aspect="auto")
    axes[2].set_title(f"Sagital (x={x_central})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(titulo, fontsize=12)
    fig.savefig(str(ruta_salida), dpi=120)
    plt.close(fig)


def figura_grilla(matriz_resultados, valores_filas, valores_columnas,
                  etiqueta_filas, etiqueta_columnas, titulo, ruta_salida):
    # Grilla NxM con el slice axial central de cada combinacion de parametros
    # matriz_resultados[i][j] = array 3D del label map para fila i, columna j
    n_filas = len(valores_filas)
    n_cols = len(valores_columnas)

    fig, axes = plt.subplots(
        n_filas, n_cols,
        figsize=(4 * n_cols, 4 * n_filas),
        constrained_layout=True,
    )
    # Asegurar que axes sea siempre 2D para poder indexar [i, j]
    if n_filas == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_filas == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, val_fila in enumerate(valores_filas):
        for j, val_col in enumerate(valores_columnas):
            ax = axes[i, j]
            arr = matriz_resultados[i][j]
            z_central = arr.shape[0] // 2
            ax.imshow(np.rot90(arr[z_central, :, :]),
                      cmap="nipy_spectral", aspect="auto")
            ax.set_title(f"{etiqueta_columnas}={val_col} {etiqueta_filas}={val_fila}",
                         fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(titulo, fontsize=14)
    fig.savefig(str(ruta_salida), dpi=120)
    plt.close(fig)


def generar_mosaico_watershed(resultados, imagen_nombre, output_dir):
    # Mosaico que reune todas las combinaciones del barrido en una sola figura.
    # Cada celda = 3 subplots horizontales (axial | coronal | sagital) sin
    # separacion interna; colormap nipy_spectral sobre el label map.
    if not resultados:
        print("[aviso] no hay resultados para generar mosaico de Watershed")
        return

    mosaics_dir = Path(output_dir) / "mosaics"
    os.makedirs(mosaics_dir, exist_ok=True)

    n_celdas = len(resultados)
    n_cols = min(MAX_CELDAS_POR_FILA, n_celdas)
    n_filas = math.ceil(n_celdas / MAX_CELDAS_POR_FILA)

    fig = plt.figure(
        figsize=(4.5 * n_cols, 2.2 * n_filas + 0.6),
        facecolor="white",
    )
    gs = GridSpec(
        n_filas, n_cols, figure=fig,
        wspace=0.05, hspace=0.25,
        top=0.88, bottom=0.02, left=0.02, right=0.98,
    )

    for idx, item in enumerate(resultados):
        fila = idx // MAX_CELDAS_POR_FILA
        columna = idx % MAX_CELDAS_POR_FILA

        # Cargar volumen .nii.gz desde disco para no acumular memoria
        try:
            volumen = itk.array_from_image(itk.imread(str(item["volumen"]), PixelType))
        except Exception as exc:
            print(f"  [aviso] no se pudo cargar {item['volumen']}: {exc}")
            continue

        nz, ny, nx = volumen.shape
        cortes = [
            np.rot90(volumen[nz // 2, :, :]),
            np.rot90(volumen[:, ny // 2, :]),
            volumen[:, :, nx // 2],
        ]
        titulo_celda = (f"s={item['sigma']} t={item['threshold']} "
                        f"l={item['level']}\n{item['tiempo']:.1f}s")

        sub_gs = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[fila, columna], wspace=0, hspace=0
        )
        for j, corte in enumerate(cortes):
            ax = fig.add_subplot(sub_gs[0, j])
            ax.imshow(corte, cmap="nipy_spectral", aspect="auto")
            ax.axis("off")
            if j == 1:
                ax.set_title(titulo_celda, fontsize=6)

    titulo_principal = f"{imagen_nombre} — Watershed — {len(resultados)} combinaciones"
    fig.suptitle(titulo_principal, fontsize=10, fontweight="bold")

    ruta_salida = mosaics_dir / f"{imagen_nombre}_watershed_mosaic.png"
    fig.savefig(str(ruta_salida), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Mosaico guardado: {ruta_salida}")


def main():
    args = parsear_argumentos()

    # Resolver rutas y crear directorios de salida
    ruta_entrada = resolver_ruta_entrada(args.input_image)
    output_dir = Path(args.output_dir)
    volumes_dir = output_dir / "volumes"
    individual_dir = output_dir / "individual"
    grids_dir = output_dir / "grids"
    gradient_dir = output_dir / "gradient"
    for d in (output_dir, volumes_dir, individual_dir, grids_dir, gradient_dir):
        os.makedirs(d, exist_ok=True)

    nombre_base = Path(args.input_image).stem
    if nombre_base.endswith(".nii"):
        nombre_base = nombre_base[:-4]

    print("=" * 70)
    print("Segmentacion por Watersheds (ITK) - Barrido de parametros")
    print("=" * 70)
    print(f"Imagen de entrada: {ruta_entrada}")
    print(f"Sigma values:     {args.sigma}")
    print(f"Threshold values: {args.threshold}")
    print(f"Level values:     {args.level}")

    # Paso 1 - Lectura: cargar la imagen medica como volumen 3D float (una sola vez)
    imagen = itk.imread(str(ruta_entrada), PixelType)
    region = imagen.GetLargestPossibleRegion()
    size = region.GetSize()
    print(f"Tamano del volumen: {size}")
    print("-" * 70)

    # Pre-calcular el gradiente para cada sigma (se reutiliza para todas las
    # combinaciones de threshold y level) - optimizacion clave del barrido
    cache_gradiente = {}
    cache_stats = {}
    for sigma in args.sigma:
        print(f"Calculando gradiente con sigma={sigma}...")
        imagen_gradiente = calcular_gradiente(imagen, sigma)
        array_g = itk.array_view_from_image(imagen_gradiente)
        g_min = float(np.min(array_g))
        g_max = float(np.max(array_g))
        g_mean = float(np.mean(array_g))
        cache_gradiente[sigma] = imagen_gradiente
        cache_stats[sigma] = (g_min, g_max, g_mean)
        # Guardar imagen de gradiente intermedia (una por sigma)
        ruta_gradiente = gradient_dir / f"{nombre_base}_gradient_s{sigma}.nii.gz"
        itk.imwrite(imagen_gradiente, str(ruta_gradiente))
        print(f"  stats: min={g_min:.6f}, max={g_max:.6f}, media={g_mean:.6f}")
        print(f"  guardado en: {ruta_gradiente}")

    # Generar todas las combinaciones de parametros
    combinaciones = list(itertools.product(args.sigma, args.threshold, args.level))
    total_combinaciones = len(combinaciones)
    print("-" * 70)
    print(f"Total de combinaciones a procesar: {total_combinaciones}")
    print("-" * 70)

    # Resultados en memoria para construir las grillas comparativas
    # resultados[(sigma, threshold, level)] = array_etiquetas
    resultados = {}
    tiempos_por_combinacion = {}
    # Lista paralela con metadatos de cada combinacion para el mosaico final
    resultados_mosaico = []

    for idx, (sigma, threshold, level) in enumerate(combinaciones, start=1):
        print(f"\n[Combinacion {idx} de {total_combinaciones}: "
              f"sigma={sigma} threshold={threshold} level={level}]")

        t_inicio = time.perf_counter()

        # Pipeline: gradiente (cacheado) -> watershed
        imagen_gradiente = cache_gradiente[sigma]
        imagen_etiquetas = aplicar_watershed(imagen_gradiente, threshold, level)

        t_total = time.perf_counter() - t_inicio
        tiempos_por_combinacion[(sigma, threshold, level)] = t_total

        array_etiquetas = itk.array_view_from_image(imagen_etiquetas).copy()
        resultados[(sigma, threshold, level)] = array_etiquetas

        # Guardar volumen .nii.gz
        nombre_arch = f"{nombre_base}_ws_s{sigma}_t{threshold}_l{level}"
        ruta_volumen = volumes_dir / f"{nombre_arch}.nii.gz"
        itk.imwrite(imagen_etiquetas, str(ruta_volumen))

        # Acumular metadatos para el mosaico final
        resultados_mosaico.append({
            "volumen": ruta_volumen,
            "sigma": sigma,
            "threshold": threshold,
            "level": level,
            "tiempo": t_total,
        })

        # Guardar figura individual (3 vistas)
        ruta_fig_indiv = individual_dir / f"{nombre_arch}.png"
        titulo = f"{nombre_base} - sigma={sigma}, threshold={threshold}, level={level}"
        figura_individual(array_etiquetas, ruta_fig_indiv, titulo)

        g_min, g_max, g_mean = cache_stats[sigma]
        print(f"  gradiente (sigma={sigma}): min={g_min:.6f}, max={g_max:.6f}, media={g_mean:.6f}")
        print(f"  volumen:  {ruta_volumen}")
        print(f"  figura:   {ruta_fig_indiv}")
        print(f"  tiempo:   {t_total:.2f} s")

    # ---------------------------------------------------------------------
    # Salida visual 2 - Grillas comparativas
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Generando grillas comparativas")
    print("=" * 70)

    # Grilla 1: threshold vs level (fijando sigma al valor medio)
    sigma_medio = args.sigma[len(args.sigma) // 2]
    matriz_g1 = [
        [resultados[(sigma_medio, t, l)] for l in args.level]
        for t in args.threshold
    ]
    ruta_g1 = grids_dir / f"{nombre_base}_grid_sigma{sigma_medio}.png"
    titulo_g1 = (f"{nombre_base} - Watershed (sigma={sigma_medio} fijo) - "
                 f"threshold (filas) vs level (columnas)")
    figura_grilla(
        matriz_g1, args.threshold, args.level,
        etiqueta_filas="t", etiqueta_columnas="l",
        titulo=titulo_g1, ruta_salida=ruta_g1,
    )
    print(f"Grilla threshold vs level: {ruta_g1}")

    # Grilla 2: sigma vs level (fijando threshold al valor medio)
    threshold_medio = args.threshold[len(args.threshold) // 2]
    matriz_g2 = [
        [resultados[(s, threshold_medio, l)] for l in args.level]
        for s in args.sigma
    ]
    ruta_g2 = grids_dir / f"{nombre_base}_grid_thresh{threshold_medio}.png"
    titulo_g2 = (f"{nombre_base} - Watershed (threshold={threshold_medio} fijo) - "
                 f"sigma (filas) vs level (columnas)")
    figura_grilla(
        matriz_g2, args.sigma, args.level,
        etiqueta_filas="s", etiqueta_columnas="l",
        titulo=titulo_g2, ruta_salida=ruta_g2,
    )
    print(f"Grilla sigma vs level:     {ruta_g2}")

    # ---------------------------------------------------------------------
    # Reporte de tiempos
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Tabla resumen de tiempos por combinacion")
    print("=" * 70)
    print(f"{'Combinacion':40s} | Tiempo (s)")
    print("-" * 55)
    total = 0.0
    for (sigma, threshold, level), t in tiempos_por_combinacion.items():
        etiq = f"s={sigma} t={threshold} l={level}"
        print(f"{etiq:40s} | {t:.2f}")
        total += t
    print("-" * 55)
    print(f"{'TOTAL':40s} | {total:.2f}")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Mosaico final con todas las combinaciones
    # ---------------------------------------------------------------------
    print("\nGenerando mosaico resumen del barrido...")
    generar_mosaico_watershed(resultados_mosaico, nombre_base, output_dir)


if __name__ == "__main__":
    main()
