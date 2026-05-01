import itk
import argparse
import os
import time
import numpy as np
from pathlib import Path

# Segmentacion por Level Sets - Fast Marching usando ITK
# Referencia: https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch4.html#x35-1890004.3
# Solo se implementa hasta la seccion 4.3.1 (Fast Marching).

# Parametros generales
Dimension = 3
PixelType = itk.F  # float, requerido por los filtros del pipeline
ImageType = itk.Image[PixelType, Dimension]
OutputPixelType = itk.UC  # uint8 para la mascara binaria final
OutputImageType = itk.Image[OutputPixelType, Dimension]


def parsear_argumentos():
    # Configuracion CLI con argparse
    parser = argparse.ArgumentParser(
        description="Segmentacion 3D por Level Sets (Fast Marching) con ITK"
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
        default=None,
        metavar=("X", "Y", "Z"),
        help="Coordenada de la semilla en indices de voxel (default: centro del volumen)",
    )
    parser.add_argument(
        "--initial_distance",
        type=float,
        default=5.0,
        help="Distancia inicial desde la semilla (default: 5.0)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Sigma del filtro de gradiente (default: 1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=-0.5,
        help="Parametro alpha del sigmoide (default: -0.5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=3.0,
        help="Parametro beta del sigmoide (default: 3.0)",
    )
    parser.add_argument(
        "--stopping_value",
        type=float,
        default=100.0,
        help="Valor de parada del Fast Marching (default: 100.0)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Iteraciones de la difusion anisotropica (default: 5)",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=0.0625,
        help="Paso de tiempo de la difusion (default: 0.0625)",
    )
    parser.add_argument(
        "--conductance",
        type=float,
        default=3.0,
        help="Conductancia de la difusion (default: 3.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/level_sets/",
        help="Directorio de salida (default: output/level_sets/)",
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


def main():
    args = parsear_argumentos()

    # Resolver rutas y crear directorios de salida
    ruta_entrada = resolver_ruta_entrada(args.input_image)
    output_dir = Path(args.output_dir)
    gradient_dir = output_dir / "gradient"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gradient_dir, exist_ok=True)

    nombre_base = Path(args.input_image).stem
    if nombre_base.endswith(".nii"):
        nombre_base = nombre_base[:-4]

    print("=" * 60)
    print("Segmentacion por Level Sets - Fast Marching (ITK)")
    print("=" * 60)
    print(f"Imagen de entrada: {ruta_entrada}")
    print(
        f"Parametros: sigma={args.sigma}, alpha={args.alpha}, beta={args.beta}, "
        f"stopping_value={args.stopping_value}"
    )
    print(
        f"Difusion: iter={args.iterations}, time_step={args.time_step}, "
        f"conductance={args.conductance}"
    )
    print("-" * 60)

    # Diccionario para almacenar tiempos por paso
    tiempos = {}

    # Paso 1 - Lectura: cargar imagen 3D float
    t0 = time.perf_counter()
    imagen = itk.imread(str(ruta_entrada), PixelType)
    region = imagen.GetLargestPossibleRegion()
    size = region.GetSize()
    tiempos["Paso 1 - Lectura"] = time.perf_counter() - t0
    print(f"Paso 1 - Lectura: tamano del volumen = {size}")

    # Determinar semilla por defecto (centro geometrico) si no fue dada
    if args.seed is None:
        seed_x = int(size[0]) // 2
        seed_y = int(size[1]) // 2
        seed_z = int(size[2]) // 2
        print(f"          semilla automatica (centro): ({seed_x}, {seed_y}, {seed_z})")
    else:
        seed_x, seed_y, seed_z = args.seed
        print(f"          semilla provista: ({seed_x}, {seed_y}, {seed_z})")

    # Paso 2 - Difusion anisotropica: suaviza la imagen preservando bordes
    # ADVERTENCIA: este filtro es el cuello de botella computacional del pipeline;
    # a mayor numberOfIterations, mayor tiempo de ejecucion;
    # valores tipicos entre 5 y 50 iteraciones.
    t0 = time.perf_counter()
    DiffusionFilter = itk.CurvatureAnisotropicDiffusionImageFilter[ImageType, ImageType]
    smoothing = DiffusionFilter.New()
    smoothing.SetInput(imagen)
    smoothing.SetNumberOfIterations(args.iterations)
    smoothing.SetTimeStep(args.time_step)
    smoothing.SetConductanceParameter(args.conductance)
    smoothing.Update()
    imagen_suavizada = smoothing.GetOutput()
    tiempos["Paso 2 - Difusion anisotropica"] = time.perf_counter() - t0
    print("Paso 2 - Difusion anisotropica: filtro aplicado")

    # Paso 3 - Gradiente: detectar bordes como zonas de alta variacion
    # sigma controla la suavidad del calculo del gradiente
    t0 = time.perf_counter()
    GradientFilter = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType]
    gradient = GradientFilter.New()
    gradient.SetInput(imagen_suavizada)
    gradient.SetSigma(args.sigma)
    gradient.Update()
    imagen_gradiente = gradient.GetOutput()
    tiempos["Paso 3 - Gradiente"] = time.perf_counter() - t0

    # Estadisticas del gradiente
    array_gradiente = itk.array_view_from_image(imagen_gradiente)
    g_min = float(np.min(array_gradiente))
    g_max = float(np.max(array_gradiente))
    g_mean = float(np.mean(array_gradiente))
    print(f"Paso 3 - Gradiente: min={g_min:.6f}, max={g_max:.6f}, media={g_mean:.6f}")

    # Guardar imagen de gradiente intermedia
    ruta_gradiente = gradient_dir / f"{nombre_base}_gradient.nii.gz"
    itk.imwrite(imagen_gradiente, str(ruta_gradiente))
    print(f"          gradiente guardado en: {ruta_gradiente}")

    # Paso 4 - Sigmoide: transformar el mapa de gradiente en una imagen de velocidad
    # alpha controla la pendiente de la sigmoide; beta el punto de inflexion;
    # juntos determinan que gradientes frenan la propagacion del frente.
    # Donde hay borde (gradiente alto), la velocidad es baja -> el frente se detiene.
    t0 = time.perf_counter()
    SigmoidFilter = itk.SigmoidImageFilter[ImageType, ImageType]
    sigmoid = SigmoidFilter.New()
    sigmoid.SetInput(imagen_gradiente)
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(args.alpha)
    sigmoid.SetBeta(args.beta)
    sigmoid.Update()
    imagen_velocidad = sigmoid.GetOutput()
    tiempos["Paso 4 - Sigmoide"] = time.perf_counter() - t0
    print("Paso 4 - Sigmoide: speed image generada")

    # Paso 5-6 - Fast Marching: propagar un frente de onda desde la semilla
    # a velocidad variable; el frente avanza rapido en regiones homogeneas
    # y lento en bordes (donde la sigmoide bajo la velocidad).
    t0 = time.perf_counter()
    NodeType = itk.LevelSetNode[PixelType, Dimension]
    NodeContainerType = itk.VectorContainer[itk.UI, NodeType]
    nodos = NodeContainerType.New()
    nodos.Initialize()

    # Construir indice ITK para la semilla
    seed_index = itk.Index[Dimension]()
    seed_index[0] = int(seed_x)
    seed_index[1] = int(seed_y)
    seed_index[2] = int(seed_z)

    # initialDistance se almacena con signo negativo siguiendo la convencion ITK
    nodo = NodeType()
    nodo.SetIndex(seed_index)
    nodo.SetValue(-args.initial_distance)
    nodos.InsertElement(0, nodo)

    FastMarchingFilter = itk.FastMarchingImageFilter[ImageType, ImageType]
    fast_marching = FastMarchingFilter.New()
    fast_marching.SetInput(imagen_velocidad)
    fast_marching.SetTrialPoints(nodos)
    fast_marching.SetStoppingValue(args.stopping_value)
    fast_marching.SetOutputSize(
        imagen_velocidad.GetBufferedRegion().GetSize()
    )
    fast_marching.Update()
    imagen_tiempo = fast_marching.GetOutput()
    tiempos["Paso 5-6 - Fast Marching"] = time.perf_counter() - t0
    print("Paso 5-6 - Fast Marching: frente propagado")

    # Paso 7 - Umbralizacion: convertir el mapa de tiempo de arribo en mascara binaria
    # lower=0, upper=stoppingValue -> los voxeles alcanzados por el frente quedan en 1
    t0 = time.perf_counter()
    ThresholdFilter = itk.BinaryThresholdImageFilter[ImageType, OutputImageType]
    threshold = ThresholdFilter.New()
    threshold.SetInput(imagen_tiempo)
    threshold.SetLowerThreshold(0.0)
    threshold.SetUpperThreshold(args.stopping_value)
    threshold.SetOutsideValue(0)
    threshold.SetInsideValue(255)
    threshold.Update()
    mascara = threshold.GetOutput()
    tiempos["Paso 7 - Umbralizacion"] = time.perf_counter() - t0
    print("Paso 7 - Umbralizacion: mascara binaria generada")

    # Guardado: escribir mascara final
    nombre_salida = (
        f"{nombre_base}_ls_seed{seed_x}_{seed_y}_{seed_z}_iter{args.iterations}.nii.gz"
    )
    ruta_salida = output_dir / nombre_salida
    itk.imwrite(mascara, str(ruta_salida))
    print(f"          mascara guardada en: {ruta_salida}")

    # Reporte de tiempo de ejecucion detallado
    total = sum(tiempos.values())
    print("-" * 60)
    print("Tiempos de ejecucion por paso:")
    for nombre, t in tiempos.items():
        print(f"  {nombre:32s} {t:8.2f} s")
    print(f"  {'TOTAL':32s} {total:8.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
