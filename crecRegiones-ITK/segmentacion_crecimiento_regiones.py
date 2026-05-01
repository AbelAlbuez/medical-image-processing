"""
Segmentación por crecimiento de regiones del ventrículo cerebral usando ITK.

Aplica dos métodos sobre el volumen A1_grayT1.nii.gz:
  - Método 1: ConnectedThresholdImageFilter   (umbrales manuales)
  - Método 2: ConfidenceConnectedImageFilter  (umbrales automáticos)

Genera los volúmenes segmentados en formato NIfTI y una figura PNG
comparativa con tres cortes axiales (z = 100, z = 110 semilla, z = 120).
"""

import os
import itk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Parámetros fijos del taller
# ---------------------------------------------------------------------------
DIR_BASE        = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_ENTRADA = os.path.join(DIR_BASE, "A1_grayT1.nii.gz")
SALIDA_MANUAL   = os.path.join(DIR_BASE, "segmentacion_manual.nii.gz")
SALIDA_AUTO     = os.path.join(DIR_BASE, "segmentacion_automatica.nii.gz")
SALIDA_FIGURA   = os.path.join(DIR_BASE, "comparacion_segmentacion.png")

# Semilla común (coordenadas en el espacio del índice ITK: x, y, z)
SEMILLA = [124, 208, 110]

# Parámetros del filtro de umbrales manuales
LOWER_MANUAL  = 7.0
UPPER_MANUAL  = 73.0

# Parámetros del filtro de umbrales automáticos
NUM_ITERACIONES = 5
MULTIPLICADOR   = 2.0
RADIO_VECINDAD  = 2

# Valor con el que se marcan los vóxeles segmentados
VALOR_SEGMENTADO = 255

# Tipos ITK de uso recurrente
DIMENSION   = 3
TipoFloat   = itk.Image[itk.F, DIMENSION]   # Tipo F para lectura/escritura
TipoEntero  = itk.Image[itk.SS, DIMENSION]  # Tipo SS requerido por Confidence


# ---------------------------------------------------------------------------
# Lectura de la imagen base
# ---------------------------------------------------------------------------
def leer_imagen_float(ruta):
    """Lee un volumen NIfTI usando pixel type itk.F."""
    lector = itk.ImageFileReader[TipoFloat].New()
    lector.SetFileName(ruta)
    lector.Update()
    return lector.GetOutput()


# ---------------------------------------------------------------------------
# Método 1: Crecimiento de regiones con umbrales manuales
# ---------------------------------------------------------------------------
def segmentar_connected_threshold(imagen):
    """Aplica ConnectedThresholdImageFilter sobre la imagen dada."""
    FiltroTipo = itk.ConnectedThresholdImageFilter[TipoFloat, TipoFloat]
    filtro = FiltroTipo.New()
    filtro.SetInput(imagen)
    filtro.SetLower(LOWER_MANUAL)
    filtro.SetUpper(UPPER_MANUAL)
    filtro.SetReplaceValue(VALOR_SEGMENTADO)
    filtro.SetSeed(SEMILLA)
    filtro.Update()
    return filtro.GetOutput()


# ---------------------------------------------------------------------------
# Método 2: Crecimiento de regiones con umbrales automáticos
# ---------------------------------------------------------------------------
def segmentar_confidence_connected(imagen):
    """Aplica ConfidenceConnectedImageFilter y convierte la salida a itk.F."""
    # El filtro requiere salida de tipo entero (SS)
    FiltroTipo = itk.ConfidenceConnectedImageFilter[TipoFloat, TipoEntero]
    filtro = FiltroTipo.New()
    filtro.SetInput(imagen)
    filtro.SetNumberOfIterations(NUM_ITERACIONES)
    filtro.SetMultiplier(MULTIPLICADOR)
    filtro.SetInitialNeighborhoodRadius(RADIO_VECINDAD)
    filtro.SetReplaceValue(VALOR_SEGMENTADO)
    filtro.SetSeed(SEMILLA)
    filtro.Update()

    # Conversión SS -> F para mantener consistencia en la escritura
    CastTipo = itk.CastImageFilter[TipoEntero, TipoFloat]
    cast = CastTipo.New()
    cast.SetInput(filtro.GetOutput())
    cast.Update()
    return cast.GetOutput()


# ---------------------------------------------------------------------------
# Escritura de un volumen NIfTI con pixel type itk.F
# ---------------------------------------------------------------------------
def guardar_imagen_float(imagen, ruta):
    """Escribe un volumen ITK de tipo float al disco."""
    escritor = itk.ImageFileWriter[TipoFloat].New()
    escritor.SetFileName(ruta)
    escritor.SetInput(imagen)
    escritor.Update()


# ---------------------------------------------------------------------------
# Visualización comparativa
# ---------------------------------------------------------------------------
def construir_figura(arr_original, arr_manual, arr_auto):
    """Genera la figura PNG con tres cortes axiales y las segmentaciones."""
    cortes = [100, 110, 120]
    titulos_cortes = [f"z = {z}" for z in cortes]
    titulos_cortes[1] += " (semilla)"

    # Colormap transparente -> color sólido (para superponer la máscara)
    cmap_cyan    = ListedColormap([(0, 0, 0, 0), (0.0, 1.0, 1.0, 0.55)])
    cmap_naranja = ListedColormap([(0, 0, 0, 0), (1.0, 0.55, 0.0, 0.55)])

    fig, ejes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(
        "Segmentación por Crecimiento de Regiones - Ventrículo Cerebral",
        fontsize=15,
        fontweight="bold",
    )

    # Fila 1: imagen original en gris
    for j, z in enumerate(cortes):
        eje = ejes[0, j]
        eje.imshow(arr_original[z], cmap="gray")
        eje.set_title(titulos_cortes[j], fontsize=11)
        eje.axis("off")
        # Marca de la semilla en el corte central
        if z == SEMILLA[2]:
            eje.plot(SEMILLA[0], SEMILLA[1], marker="+",
                     color="red", markersize=14, markeredgewidth=2)

    # Fila 2: original + segmentación manual (cian)
    for j, z in enumerate(cortes):
        eje = ejes[1, j]
        eje.imshow(arr_original[z], cmap="gray")
        mascara = (arr_manual[z] > 0).astype(np.uint8)
        eje.imshow(mascara, cmap=cmap_cyan, vmin=0, vmax=1)
        eje.set_title(titulos_cortes[j], fontsize=11)
        eje.axis("off")

    # Fila 3: original + segmentación automática (naranja)
    for j, z in enumerate(cortes):
        eje = ejes[2, j]
        eje.imshow(arr_original[z], cmap="gray")
        mascara = (arr_auto[z] > 0).astype(np.uint8)
        eje.imshow(mascara, cmap=cmap_naranja, vmin=0, vmax=1)
        eje.set_title(titulos_cortes[j], fontsize=11)
        eje.axis("off")

    # Subtítulos por fila (a la izquierda de cada fila)
    ejes[1, 0].annotate(
        "ConnectedThreshold [Lower=7, Upper=73]",
        xy=(-0.15, 0.5), xycoords="axes fraction",
        rotation=90, ha="center", va="center",
        fontsize=11, fontweight="bold", color="teal",
    )
    ejes[2, 0].annotate(
        "ConfidenceConnected [Mult=2.0, Iter=5, Radio=2]",
        xy=(-0.15, 0.5), xycoords="axes fraction",
        rotation=90, ha="center", va="center",
        fontsize=11, fontweight="bold", color="darkorange",
    )

    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    plt.savefig(SALIDA_FIGURA, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    print("==> Leyendo volumen de entrada:", ARCHIVO_ENTRADA)
    imagen = leer_imagen_float(ARCHIVO_ENTRADA)

    print("==> Ejecutando ConnectedThresholdImageFilter (manual)")
    seg_manual = segmentar_connected_threshold(imagen)
    guardar_imagen_float(seg_manual, SALIDA_MANUAL)
    print("    Guardado:", SALIDA_MANUAL)

    print("==> Ejecutando ConfidenceConnectedImageFilter (automático)")
    seg_auto = segmentar_confidence_connected(imagen)
    guardar_imagen_float(seg_auto, SALIDA_AUTO)
    print("    Guardado:", SALIDA_AUTO)

    # Conversión a numpy: ITK devuelve arreglos en orden (z, y, x)
    arr_original = itk.GetArrayViewFromImage(imagen)
    arr_manual   = itk.GetArrayViewFromImage(seg_manual)
    arr_auto     = itk.GetArrayViewFromImage(seg_auto)

    print("==> Generando figura comparativa")
    construir_figura(arr_original, arr_manual, arr_auto)
    print("    Guardada:", SALIDA_FIGURA)

    # Conteo de vóxeles segmentados por cada método
    voxeles_manual = int(np.count_nonzero(arr_manual))
    voxeles_auto   = int(np.count_nonzero(arr_auto))
    print("---------------------------------------------------------")
    print(f"Vóxeles segmentados ConnectedThreshold : {voxeles_manual}")
    print(f"Vóxeles segmentados ConfidenceConnected: {voxeles_auto}")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    main()
