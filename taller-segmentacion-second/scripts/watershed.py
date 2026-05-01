import itk
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Segmentacion por Watersheds usando ITK
# Referencia: https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch4.html#x35-1860004.2

# Parametros generales
Dimension = 3
PixelType = itk.F  # float, requerido por GradientMagnitudeRecursiveGaussian
ImageType = itk.Image[PixelType, Dimension]


def parsear_argumentos():
    # Configuracion CLI con argparse
    parser = argparse.ArgumentParser(
        description="Segmentacion 3D por Watersheds con ITK"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Ruta al archivo de entrada (.nii.gz). Si es relativa, se resuelve contra ./images/",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Sigma del filtro de gradiente gaussiano (default: 1.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Umbral minimo de gradiente considerado como borde (default: 0.01)",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.2,
        help="Nivel de inundacion - controla la fusion de cuencas (default: 0.2)",
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


def main():
    args = parsear_argumentos()

    # Iniciar cronometro global
    t_inicio = time.perf_counter()

    # Resolver rutas y crear directorios de salida
    ruta_entrada = resolver_ruta_entrada(args.input_image)
    output_dir = Path(args.output_dir)
    gradient_dir = output_dir / "gradient"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gradient_dir, exist_ok=True)

    nombre_base = Path(args.input_image).stem
    # Quitar extension .nii si quedo despues del .gz
    if nombre_base.endswith(".nii"):
        nombre_base = nombre_base[:-4]

    print("=" * 60)
    print("Segmentacion por Watersheds (ITK)")
    print("=" * 60)
    print(f"Imagen de entrada: {ruta_entrada}")
    print(f"Parametros: sigma={args.sigma}, threshold={args.threshold}, level={args.level}")
    print("-" * 60)

    # Paso 1 - Lectura: cargar la imagen medica como volumen 3D float
    imagen = itk.imread(str(ruta_entrada), PixelType)
    region = imagen.GetLargestPossibleRegion()
    size = region.GetSize()
    print(f"Paso 1 - Lectura: tamano del volumen = {size}")

    # Paso 2 - Gradiente: calcular la magnitud del gradiente gaussiano
    # Los bordes son zonas de alto gradiente; sigma controla la suavidad del calculo
    GradientFilter = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType]
    gradient = GradientFilter.New()
    gradient.SetInput(imagen)
    gradient.SetSigma(args.sigma)
    gradient.Update()
    imagen_gradiente = gradient.GetOutput()

    # Estadisticas del gradiente para ayudar a elegir threshold y level
    array_gradiente = itk.array_view_from_image(imagen_gradiente)
    g_min = float(np.min(array_gradiente))
    g_max = float(np.max(array_gradiente))
    g_mean = float(np.mean(array_gradiente))
    print(f"Paso 2 - Gradiente: min={g_min:.6f}, max={g_max:.6f}, media={g_mean:.6f}")

    # Guardar imagen de gradiente intermedia
    ruta_gradiente = gradient_dir / f"{nombre_base}_gradient.nii.gz"
    itk.imwrite(imagen_gradiente, str(ruta_gradiente))
    print(f"          gradiente guardado en: {ruta_gradiente}")

    # Paso 3 - Watershed: inundar el mapa de gradiente desde los minimos locales
    # threshold descarta gradientes muy bajos (ruido)
    # level controla hasta que altura se inunda antes de detener la fusion de cuencas
    WatershedFilter = itk.WatershedImageFilter[ImageType]
    watershed = WatershedFilter.New()
    watershed.SetInput(imagen_gradiente)
    watershed.SetThreshold(args.threshold)
    watershed.SetLevel(args.level)
    watershed.Update()
    imagen_etiquetas = watershed.GetOutput()
    print("Paso 3 - Watershed: filtro aplicado")

    # Paso 4 - Guardado: escribir label map
    nombre_salida = (
        f"{nombre_base}_ws_s{args.sigma}_t{args.threshold}_l{args.level}.nii.gz"
    )
    ruta_salida = output_dir / nombre_salida
    itk.imwrite(imagen_etiquetas, str(ruta_salida))
    print(f"Paso 4 - Guardado: label map en {ruta_salida}")

    # Vista previa rapida del label map
    # NOTA: el WatershedImageFilter produce un label map con muchas etiquetas
    # (oversegmentation es normal). El colormap es esencial para interpretar el
    # resultado: sin color, las regiones adyacentes son indistinguibles porque
    # los valores de etiqueta consecutivos lucen casi identicos en escala de grises.
    array_etiquetas = itk.array_view_from_image(imagen_etiquetas)
    z_central = array_etiquetas.shape[0] // 2
    y_central = array_etiquetas.shape[1] // 2
    x_central = array_etiquetas.shape[2] // 2

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
    fig.suptitle(
        f"Watershed - sigma={args.sigma}, threshold={args.threshold}, level={args.level}",
        fontsize=12,
    )
    ruta_figura = output_dir / f"{nombre_base}_ws_preview.png"
    fig.savefig(str(ruta_figura), dpi=120)
    plt.close(fig)
    print(f"          vista previa en {ruta_figura}")

    # Reporte de tiempo total
    t_total = time.perf_counter() - t_inicio
    print("-" * 60)
    print(f"Tiempo total de ejecucion: {t_total:.2f} segundos")
    print("=" * 60)


if __name__ == "__main__":
    main()
