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

# Segmentacion por Level Sets - Fast Marching usando ITK - Barrido en 2 fases
# Referencia: https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch4.html#x35-1890004.3
# Solo se implementa hasta la seccion 4.3.1 (Fast Marching).

# Parametros generales
Dimension = 3
PixelType = itk.F  # float, requerido por los filtros del pipeline
ImageType = itk.Image[PixelType, Dimension]
OutputPixelType = itk.UC  # uint8 para la mascara binaria final
OutputImageType = itk.Image[OutputPixelType, Dimension]

# ---------------------------------------------------------------------------
# FASE 1 - barrido de iterations + sigma con sigmoide y stoppingValue fijos
# ---------------------------------------------------------------------------
ALPHA_FIXED    = -0.5
BETA_FIXED     = 3.0
STOPPING_FIXED = 100.0

ITERATIONS_VALUES = [5, 15, 30]
SIGMA_VALUES      = [0.5, 1.0, 2.0]

# ---------------------------------------------------------------------------
# FASE 2 - barrido de alpha + beta + stoppingValue con iter y sigma fijos
# Editar ITERATIONS_BEST y SIGMA_BEST tras revisar los resultados de la fase 1
# ---------------------------------------------------------------------------
ITERATIONS_BEST = 15
SIGMA_BEST      = 1.0

ALPHA_VALUES    = [-0.5, -1.0, -2.0]
BETA_VALUES     = [2.0, 3.0, 5.0]
STOPPING_VALUES = [50.0, 100.0, 200.0]

# Difusion anisotropica - parametros adicionales no barridos
TIME_STEP_DEFAULT   = 0.0625
CONDUCTANCE_DEFAULT = 3.0


def parsear_argumentos():
    # Configuracion CLI con argparse
    parser = argparse.ArgumentParser(
        description="Segmentacion 3D por Level Sets (Fast Marching) con ITK - barrido en 2 fases"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Ruta al archivo de entrada (.nii.gz). Si es relativa, se resuelve contra ./images/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Coordenada de la semilla en indices de voxel (requerido)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Fase del barrido (1: iter+sigma, 2: alpha+beta+stopping). Default: 1",
    )
    parser.add_argument(
        "--initial_distance",
        type=float,
        default=5.0,
        help="Distancia inicial desde la semilla (default: 5.0)",
    )
    # Argumentos de fase 1 (multiples valores)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=ITERATIONS_VALUES,
        help=f"Uno o mas valores de iteraciones de difusion (default fase 1: {ITERATIONS_VALUES})",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=SIGMA_VALUES,
        help=f"Uno o mas valores de sigma del gradiente (default fase 1: {SIGMA_VALUES})",
    )
    # Argumentos de fase 2 (multiples valores)
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=ALPHA_VALUES,
        help=f"Uno o mas valores de alpha del sigmoide (default fase 2: {ALPHA_VALUES})",
    )
    parser.add_argument(
        "--beta",
        type=float,
        nargs="+",
        default=BETA_VALUES,
        help=f"Uno o mas valores de beta del sigmoide (default fase 2: {BETA_VALUES})",
    )
    parser.add_argument(
        "--stopping",
        type=float,
        nargs="+",
        default=STOPPING_VALUES,
        help=f"Uno o mas valores de stoppingValue (default fase 2: {STOPPING_VALUES})",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=TIME_STEP_DEFAULT,
        help=f"Paso de tiempo de la difusion (default: {TIME_STEP_DEFAULT})",
    )
    parser.add_argument(
        "--conductance",
        type=float,
        default=CONDUCTANCE_DEFAULT,
        help=f"Conductancia de la difusion (default: {CONDUCTANCE_DEFAULT})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/level_sets/",
        help="Directorio raiz de salida (default: output/level_sets/)",
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


def aplicar_difusion(imagen, iterations, time_step, conductance):
    # Paso 2 - Difusion anisotropica: suaviza preservando bordes
    # ADVERTENCIA: cuello de botella computacional del pipeline
    DiffusionFilter = itk.CurvatureAnisotropicDiffusionImageFilter[ImageType, ImageType]
    smoothing = DiffusionFilter.New()
    smoothing.SetInput(imagen)
    smoothing.SetNumberOfIterations(iterations)
    smoothing.SetTimeStep(time_step)
    smoothing.SetConductanceParameter(conductance)
    smoothing.Update()
    return smoothing.GetOutput()


def aplicar_gradiente(imagen, sigma):
    # Paso 3 - Gradiente: zonas de alta variacion son bordes
    GradientFilter = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType]
    gradient = GradientFilter.New()
    gradient.SetInput(imagen)
    gradient.SetSigma(sigma)
    gradient.Update()
    return gradient.GetOutput()


def aplicar_sigmoide(imagen_gradiente, alpha, beta):
    # Paso 4 - Sigmoide: convierte gradiente en speed image
    # alpha controla la pendiente; beta el punto de inflexion
    # gradiente alto -> velocidad baja -> el frente se detiene
    SigmoidFilter = itk.SigmoidImageFilter[ImageType, ImageType]
    sigmoid = SigmoidFilter.New()
    sigmoid.SetInput(imagen_gradiente)
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(alpha)
    sigmoid.SetBeta(beta)
    sigmoid.Update()
    return sigmoid.GetOutput()


def aplicar_fast_marching(imagen_velocidad, seed, initial_distance, stopping_value):
    # Paso 5-6 - Fast Marching: propagar el frente desde la semilla
    NodeType = itk.LevelSetNode[PixelType, Dimension]
    NodeContainerType = itk.VectorContainer[itk.UI, NodeType]
    nodos = NodeContainerType.New()
    nodos.Initialize()

    seed_index = itk.Index[Dimension]()
    seed_index[0] = int(seed[0])
    seed_index[1] = int(seed[1])
    seed_index[2] = int(seed[2])

    # initialDistance se almacena con signo negativo siguiendo la convencion ITK
    nodo = NodeType()
    nodo.SetIndex(seed_index)
    nodo.SetValue(-initial_distance)
    nodos.InsertElement(0, nodo)

    FastMarchingFilter = itk.FastMarchingImageFilter[ImageType, ImageType]
    fast_marching = FastMarchingFilter.New()
    fast_marching.SetInput(imagen_velocidad)
    fast_marching.SetTrialPoints(nodos)
    fast_marching.SetStoppingValue(stopping_value)
    fast_marching.SetOutputSize(imagen_velocidad.GetBufferedRegion().GetSize())
    fast_marching.Update()
    return fast_marching.GetOutput()


def aplicar_umbralizacion(imagen_tiempo, stopping_value):
    # Paso 7 - Umbralizacion: tiempo de arribo -> mascara binaria
    ThresholdFilter = itk.BinaryThresholdImageFilter[ImageType, OutputImageType]
    threshold = ThresholdFilter.New()
    threshold.SetInput(imagen_tiempo)
    threshold.SetLowerThreshold(0.0)
    threshold.SetUpperThreshold(stopping_value)
    threshold.SetOutsideValue(0)
    threshold.SetInsideValue(255)
    threshold.Update()
    return threshold.GetOutput()


def figura_individual(array_original, array_mascara, ruta_salida, titulo):
    # Genera figura 1x3 con cortes axial, coronal y sagital de la imagen original
    # con el contorno rojo de la segmentacion superpuesto.
    # El contorno rojo es clave para ver la segmentacion sobre la anatomia original.
    nz, ny, nx = array_original.shape
    z_central = nz // 2
    y_central = ny // 2
    x_central = nx // 2

    vmax = float(np.percentile(array_original, 98))

    cortes = [
        ("Axial",   np.rot90(array_original[z_central, :, :]),
                    np.rot90(array_mascara[z_central, :, :]),  z_central, "z"),
        ("Coronal", np.rot90(array_original[:, y_central, :]),
                    np.rot90(array_mascara[:, y_central, :]),  y_central, "y"),
        ("Sagital", array_original[:, :, x_central],
                    array_mascara[:, :, x_central],            x_central, "x"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (nombre_plano, corte_orig, corte_mask, idx, eje) in zip(axes, cortes):
        ax.imshow(corte_orig, cmap="gray", vmin=0, vmax=vmax, aspect="auto")
        # Contorno rojo de la mascara binaria sobre la anatomia original
        if np.any(corte_mask > 0):
            ax.contour(corte_mask > 0, levels=[0.5], colors="red", linewidths=1.0)
        ax.set_title(f"{nombre_plano} ({eje}={idx})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(titulo, fontsize=12)
    fig.savefig(str(ruta_salida), dpi=120)
    plt.close(fig)


def figura_grilla(matriz_mascaras, array_original, valores_filas, valores_columnas,
                  etiqueta_filas, etiqueta_columnas, titulo, ruta_salida):
    # Grilla NxM con slice axial central de la imagen original y contorno rojo
    # de la segmentacion superpuesto, una celda por combinacion de parametros.
    n_filas = len(valores_filas)
    n_cols = len(valores_columnas)

    z_central = array_original.shape[0] // 2
    corte_original = np.rot90(array_original[z_central, :, :])
    vmax = float(np.percentile(array_original, 98))

    fig, axes = plt.subplots(
        n_filas, n_cols,
        figsize=(4 * n_cols, 4 * n_filas),
        constrained_layout=True,
    )
    if n_filas == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_filas == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, val_fila in enumerate(valores_filas):
        for j, val_col in enumerate(valores_columnas):
            ax = axes[i, j]
            mascara = matriz_mascaras[i][j]
            corte_mask = np.rot90(mascara[z_central, :, :])
            ax.imshow(corte_original, cmap="gray", vmin=0, vmax=vmax, aspect="auto")
            if np.any(corte_mask > 0):
                ax.contour(corte_mask > 0, levels=[0.5], colors="red", linewidths=1.0)
            ax.set_title(f"{etiqueta_filas}={val_fila} {etiqueta_columnas}={val_col}",
                         fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(titulo, fontsize=14)
    fig.savefig(str(ruta_salida), dpi=120)
    plt.close(fig)


def imprimir_tiempos(etiq_combinacion, tiempos):
    # Imprime la tabla de 6 pasos + total para una combinacion
    print(f"  [{etiq_combinacion}]")
    for nombre, t in tiempos.items():
        print(f"    {nombre:32s} {t:8.2f} s")
    total = sum(tiempos.values())
    print(f"    {'TOTAL combinacion':32s} {total:8.2f} s")
    return total


def generar_mosaico_level_sets(resultados, imagen_nombre, imagen_original_path,
                                fase, output_dir):
    # Mosaico que reune todas las combinaciones de la fase en una sola figura.
    # Cada celda = 3 subplots (axial | coronal | sagital) con la imagen original
    # en gris de fondo y el contorno rojo de la mascara superpuesto.
    if not resultados:
        print(f"[aviso] no hay resultados para generar mosaico de fase {fase}")
        return

    mosaics_dir = Path(output_dir) / "mosaics"
    os.makedirs(mosaics_dir, exist_ok=True)

    # Cargar imagen original (fondo gris) una sola vez
    array_original = None
    if imagen_original_path and Path(imagen_original_path).exists():
        try:
            array_original = itk.array_from_image(
                itk.imread(str(imagen_original_path), PixelType)
            )
        except Exception as exc:
            print(f"[aviso] no se pudo cargar imagen original {imagen_original_path}: {exc}")
            array_original = None
    if array_original is None:
        print(f"[aviso] sin imagen original; el contorno rojo necesita la anatomia de fondo")

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
        top=0.86, bottom=0.02, left=0.02, right=0.98,
    )

    vmax = float(np.percentile(array_original, 98)) if array_original is not None else 1.0

    for idx, item in enumerate(resultados):
        fila = idx // MAX_CELDAS_POR_FILA
        columna = idx % MAX_CELDAS_POR_FILA

        # Cargar mascara binaria desde disco
        try:
            mascara = itk.array_from_image(itk.imread(str(item["volumen"]), PixelType))
        except Exception as exc:
            print(f"  [aviso] no se pudo cargar {item['volumen']}: {exc}")
            continue

        nz, ny, nx = mascara.shape
        cortes_mask = [
            np.rot90(mascara[nz // 2, :, :]),
            np.rot90(mascara[:, ny // 2, :]),
            mascara[:, :, nx // 2],
        ]
        if array_original is not None:
            cortes_orig = [
                np.rot90(array_original[nz // 2, :, :]),
                np.rot90(array_original[:, ny // 2, :]),
                array_original[:, :, nx // 2],
            ]
        else:
            cortes_orig = [None, None, None]

        # Titulo de celda segun fase
        if fase == 1:
            titulo_celda = (f"iter={item['iterations']} σ={item['sigma']}\n"
                            f"total={item['tiempo_total']:.1f}s "
                            f"dif={item['tiempo_difusion']:.1f}s")
        else:
            titulo_celda = (f"α={item['alpha']} β={item['beta']} stop={item['stopping']}\n"
                            f"total={item['tiempo_total']:.1f}s")

        sub_gs = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[fila, columna], wspace=0, hspace=0
        )
        for j in range(3):
            ax = fig.add_subplot(sub_gs[0, j])
            if cortes_orig[j] is not None:
                ax.imshow(cortes_orig[j], cmap="gray", vmin=0, vmax=vmax,
                          aspect="auto")
            if np.any(cortes_mask[j] > 0):
                ax.contour(cortes_mask[j] > 0, levels=[0.5],
                           colors="red", linewidths=0.8)
            ax.axis("off")
            if j == 1:
                ax.set_title(titulo_celda, fontsize=6)

    titulo_principal = (f"{imagen_nombre} — Level Sets Fase {fase} — "
                        f"{len(resultados)} combinaciones")
    fig.suptitle(titulo_principal, fontsize=10, fontweight="bold")

    if fase == 1:
        subtitulo = (f"Fijos: alpha={ALPHA_FIXED} beta={BETA_FIXED} "
                     f"stopping={STOPPING_FIXED}")
    else:
        subtitulo = f"Fijos: iterations={ITERATIONS_BEST} sigma={SIGMA_BEST}"
    fig.text(0.5, 0.92, subtitulo, ha="center", va="top", fontsize=8)

    ruta_salida = mosaics_dir / f"{imagen_nombre}_ls_fase{fase}_mosaic.png"
    fig.savefig(str(ruta_salida), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Mosaico guardado: {ruta_salida}")


def main():
    args = parsear_argumentos()

    # Resolver rutas y crear directorios de salida (organizados por fase)
    ruta_entrada = resolver_ruta_entrada(args.input_image)
    output_root = Path(args.output_dir)
    fase_dir = output_root / f"fase{args.phase}"
    volumes_dir = fase_dir / "volumes"
    individual_dir = fase_dir / "individual"
    grids_dir = fase_dir / "grids"
    gradient_dir = output_root / "gradient"
    for d in (output_root, fase_dir, volumes_dir, individual_dir, grids_dir, gradient_dir):
        os.makedirs(d, exist_ok=True)

    nombre_base = Path(args.input_image).stem
    if nombre_base.endswith(".nii"):
        nombre_base = nombre_base[:-4]

    print("=" * 70)
    print(f"Segmentacion por Level Sets - Fast Marching (ITK) - FASE {args.phase}")
    print("=" * 70)
    print(f"Imagen de entrada: {ruta_entrada}")
    print(f"Semilla: ({args.seed[0]}, {args.seed[1]}, {args.seed[2]})")

    # Paso 1 - Lectura: cargar imagen 3D float (una sola vez)
    t0 = time.perf_counter()
    imagen = itk.imread(str(ruta_entrada), PixelType)
    region = imagen.GetLargestPossibleRegion()
    size = region.GetSize()
    tiempo_lectura = time.perf_counter() - t0
    print(f"Tamano del volumen: {size}")
    print(f"Tiempo de lectura inicial: {tiempo_lectura:.2f} s")

    array_original = itk.array_view_from_image(imagen).copy()

    # ---------------------------------------------------------------------
    # Construir el conjunto de combinaciones segun la fase
    # ---------------------------------------------------------------------
    if args.phase == 1:
        # Fase 1: variables = iterations, sigma; fijos = alpha, beta, stopping
        valores_iter = args.iterations
        valores_sigma = args.sigma
        combinaciones = list(itertools.product(valores_iter, valores_sigma))
        print(f"Iterations:  {valores_iter}")
        print(f"Sigma:       {valores_sigma}")
        print(f"Fijos: alpha={ALPHA_FIXED}, beta={BETA_FIXED}, stopping={STOPPING_FIXED}")
    else:
        # Fase 2: variables = alpha, beta, stopping; fijos = iter, sigma
        valores_alpha = args.alpha
        valores_beta = args.beta
        valores_stopping = args.stopping
        combinaciones = list(itertools.product(valores_alpha, valores_beta, valores_stopping))
        print(f"Alpha:       {valores_alpha}")
        print(f"Beta:        {valores_beta}")
        print(f"Stopping:    {valores_stopping}")
        print(f"Fijos: iterations={ITERATIONS_BEST}, sigma={SIGMA_BEST}")

    total_combinaciones = len(combinaciones)
    print("-" * 70)
    print(f"Total de combinaciones a procesar: {total_combinaciones}")
    print("-" * 70)

    # Para fase 2 podemos pre-calcular difusion + gradiente una sola vez,
    # ya que iterations y sigma estan fijos
    cache_velocidad_base = None
    cache_imagen_gradiente = None
    cache_array_gradiente = None
    if args.phase == 2:
        print("Pre-calculando difusion y gradiente (parametros fijos en fase 2)...")
        imagen_suavizada = aplicar_difusion(
            imagen, ITERATIONS_BEST, args.time_step, args.conductance
        )
        cache_imagen_gradiente = aplicar_gradiente(imagen_suavizada, SIGMA_BEST)
        cache_array_gradiente = itk.array_view_from_image(cache_imagen_gradiente)
        ruta_g = gradient_dir / f"{nombre_base}_gradient_iter{ITERATIONS_BEST}_s{SIGMA_BEST}.nii.gz"
        itk.imwrite(cache_imagen_gradiente, str(ruta_g))
        g_min = float(np.min(cache_array_gradiente))
        g_max = float(np.max(cache_array_gradiente))
        g_mean = float(np.mean(cache_array_gradiente))
        print(f"  gradiente: min={g_min:.6f}, max={g_max:.6f}, media={g_mean:.6f}")
        print(f"  guardado en: {ruta_g}")
        print("-" * 70)

    # Resultados en memoria para construir las grillas
    resultados = {}
    tiempos_resumen = {}  # combinacion -> (tiempo_difusion, tiempo_total)
    # Lista paralela con metadatos de cada combinacion para el mosaico final
    resultados_mosaico = []

    for idx, combinacion in enumerate(combinaciones, start=1):
        if args.phase == 1:
            iterations, sigma = combinacion
            alpha, beta, stopping = ALPHA_FIXED, BETA_FIXED, STOPPING_FIXED
            etiq = f"Fase 1 | iter={iterations} sigma={sigma}"
            nombre_arch = (f"{nombre_base}_ls_iter{iterations}_s{sigma}_"
                           f"a{alpha}_b{beta}_st{stopping}")
        else:
            alpha, beta, stopping = combinacion
            iterations, sigma = ITERATIONS_BEST, SIGMA_BEST
            etiq = f"Fase 2 | alpha={alpha} beta={beta} stopping={stopping}"
            nombre_arch = (f"{nombre_base}_ls_iter{iterations}_s{sigma}_"
                           f"a{alpha}_b{beta}_st{stopping}")

        print(f"\n[Combinacion {idx} de {total_combinaciones}: {etiq}]")

        tiempos = {}

        # Paso 1 - Lectura ya hecha; se anota su tiempo prorrateado
        tiempos["Paso 1 - Lectura"] = tiempo_lectura

        if args.phase == 1:
            # Paso 2 - Difusion anisotropica (varia por iter)
            t0 = time.perf_counter()
            imagen_suavizada = aplicar_difusion(
                imagen, iterations, args.time_step, args.conductance
            )
            tiempos["Paso 2 - Difusion anisotropica"] = time.perf_counter() - t0

            # Paso 3 - Gradiente (varia por sigma)
            t0 = time.perf_counter()
            imagen_gradiente = aplicar_gradiente(imagen_suavizada, sigma)
            tiempos["Paso 3 - Gradiente"] = time.perf_counter() - t0

            # Estadisticas y guardado del gradiente (uno por combinacion)
            array_g = itk.array_view_from_image(imagen_gradiente)
            g_min = float(np.min(array_g))
            g_max = float(np.max(array_g))
            g_mean = float(np.mean(array_g))
            ruta_g = gradient_dir / f"{nombre_base}_gradient_iter{iterations}_s{sigma}.nii.gz"
            itk.imwrite(imagen_gradiente, str(ruta_g))
            print(f"    gradiente: min={g_min:.6f}, max={g_max:.6f}, media={g_mean:.6f}")
        else:
            # Difusion y gradiente cacheados
            tiempos["Paso 2 - Difusion anisotropica"] = 0.0
            tiempos["Paso 3 - Gradiente"] = 0.0
            imagen_gradiente = cache_imagen_gradiente

        # Paso 4 - Sigmoide
        t0 = time.perf_counter()
        imagen_velocidad = aplicar_sigmoide(imagen_gradiente, alpha, beta)
        tiempos["Paso 4 - Sigmoide"] = time.perf_counter() - t0

        # Paso 5-6 - Fast Marching
        t0 = time.perf_counter()
        imagen_tiempo = aplicar_fast_marching(
            imagen_velocidad, args.seed, args.initial_distance, stopping
        )
        tiempos["Paso 5-6 - Fast Marching"] = time.perf_counter() - t0

        # Paso 7 - Umbralizacion
        t0 = time.perf_counter()
        mascara = aplicar_umbralizacion(imagen_tiempo, stopping)
        tiempos["Paso 7 - Umbralizacion"] = time.perf_counter() - t0

        # Tabla de tiempos por combinacion
        total_combo = imprimir_tiempos(etiq, tiempos)
        tiempos_resumen[combinacion] = (
            tiempos["Paso 2 - Difusion anisotropica"], total_combo
        )

        # Guardado del volumen
        ruta_volumen = volumes_dir / f"{nombre_arch}.nii.gz"
        itk.imwrite(mascara, str(ruta_volumen))

        # Figura individual con contorno rojo
        array_mascara = itk.array_view_from_image(mascara).copy()
        ruta_fig = individual_dir / f"{nombre_arch}.png"
        titulo = f"{nombre_base} - {etiq}"
        figura_individual(array_original, array_mascara, ruta_fig, titulo)

        resultados[combinacion] = array_mascara

        # Acumular metadatos para el mosaico final
        item_mosaico = {
            "volumen": ruta_volumen,
            "tiempo_total": total_combo,
            "tiempo_difusion": tiempos["Paso 2 - Difusion anisotropica"],
        }
        if args.phase == 1:
            item_mosaico["iterations"] = iterations
            item_mosaico["sigma"] = sigma
        else:
            item_mosaico["alpha"] = alpha
            item_mosaico["beta"] = beta
            item_mosaico["stopping"] = stopping
        resultados_mosaico.append(item_mosaico)

        print(f"    volumen:  {ruta_volumen}")
        print(f"    figura:   {ruta_fig}")

    # ---------------------------------------------------------------------
    # Salida visual 2 - Grillas comparativas
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Generando grillas comparativas")
    print("=" * 70)

    if args.phase == 1:
        # Grilla iterations vs sigma
        matriz = [
            [resultados[(it, s)] for s in args.sigma]
            for it in args.iterations
        ]
        ruta_grilla = grids_dir / f"{nombre_base}_grid_fase1.png"
        titulo = (f"{nombre_base} - Level Sets fase 1 - "
                  f"iterations (filas) vs sigma (columnas) | "
                  f"alpha={ALPHA_FIXED}, beta={BETA_FIXED}, stopping={STOPPING_FIXED}")
        figura_grilla(
            matriz, array_original, args.iterations, args.sigma,
            etiqueta_filas="iter", etiqueta_columnas="s",
            titulo=titulo, ruta_salida=ruta_grilla,
        )
        print(f"Grilla iterations vs sigma: {ruta_grilla}")

    else:
        # Grilla 1 fase 2: alpha vs beta (fijando stopping al valor medio)
        stopping_medio = args.stopping[len(args.stopping) // 2]
        matriz_ab = [
            [resultados[(a, b, stopping_medio)] for b in args.beta]
            for a in args.alpha
        ]
        ruta_ab = grids_dir / f"{nombre_base}_grid_alpha_beta.png"
        titulo_ab = (f"{nombre_base} - Level Sets fase 2 (stopping={stopping_medio} fijo) - "
                     f"alpha (filas) vs beta (columnas)")
        figura_grilla(
            matriz_ab, array_original, args.alpha, args.beta,
            etiqueta_filas="a", etiqueta_columnas="b",
            titulo=titulo_ab, ruta_salida=ruta_ab,
        )
        print(f"Grilla alpha vs beta:    {ruta_ab}")

        # Grilla 2 fase 2: stopping vs beta (fijando alpha al valor medio)
        alpha_medio = args.alpha[len(args.alpha) // 2]
        matriz_sb = [
            [resultados[(alpha_medio, b, st)] for b in args.beta]
            for st in args.stopping
        ]
        ruta_sb = grids_dir / f"{nombre_base}_grid_stopping_beta.png"
        titulo_sb = (f"{nombre_base} - Level Sets fase 2 (alpha={alpha_medio} fijo) - "
                     f"stopping (filas) vs beta (columnas)")
        figura_grilla(
            matriz_sb, array_original, args.stopping, args.beta,
            etiqueta_filas="st", etiqueta_columnas="b",
            titulo=titulo_sb, ruta_salida=ruta_sb,
        )
        print(f"Grilla stopping vs beta: {ruta_sb}")

    # ---------------------------------------------------------------------
    # Tabla resumen final por combinacion
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Tabla resumen de tiempos por combinacion")
    print("=" * 70)
    print(f"{'Combinacion':40s} | {'Difusion (s)':>12s} | {'Total (s)':>10s}")
    print("-" * 70)
    for combo, (t_dif, t_tot) in tiempos_resumen.items():
        if args.phase == 1:
            it, s = combo
            etiq = f"iter={it} sigma={s}"
        else:
            a, b, st = combo
            etiq = f"alpha={a} beta={b} stop={st}"
        print(f"{etiq:40s} | {t_dif:12.2f} | {t_tot:10.2f}")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Mosaico final con todas las combinaciones de la fase actual
    # ---------------------------------------------------------------------
    print(f"\nGenerando mosaico resumen de la fase {args.phase}...")
    generar_mosaico_level_sets(
        resultados_mosaico,
        nombre_base,
        ruta_entrada,
        args.phase,
        fase_dir,
    )


if __name__ == "__main__":
    main()
