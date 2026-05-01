import itk
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path

# Generacion automatica de mosaicos comparativos para todas las combinaciones
# (imagen, metodo) que tengan resultados en output/. Sin argumentos CLI.

# ---------------------------------------------------------------------------
# Configuracion fija
# ---------------------------------------------------------------------------
IMAGENES = ["MRBrainTumor", "MRBreastCancer", "MRLiverTumor"]

METODOS = {
    "watershed":        "output/watershed/volumes/",
    "level_sets_fase1": "output/level_sets/fase1/volumes/",
    "level_sets_fase2": "output/level_sets/fase2/volumes/",
}

IMAGES_DIR = Path("images")
OUTPUT_MOSAICS_DIR = Path("output/mosaics")
MAX_CELDAS_POR_FILA = 4

# Subtitulos con parametros fijos por metodo (None = sin subtitulo)
SUBTITULOS_FIJOS = {
    "watershed":        None,
    "level_sets_fase1": "Fijos: alpha=-0.5, beta=3.0, stopping=100.0",
    "level_sets_fase2": "Fijos: iterations=ITERATIONS_BEST, sigma=SIGMA_BEST",
}

# Parametros generales ITK
Dimension = 3
PixelType = itk.F


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def leer_volumen(ruta):
    # Carga un .nii.gz como float y devuelve el array numpy en orden (Z, Y, X).
    # Si falla, devuelve None y deja que el llamador continue.
    try:
        imagen = itk.imread(str(ruta), PixelType)
        return itk.array_from_image(imagen)
    except Exception as exc:
        print(f"  [aviso] no se pudo cargar {ruta}: {exc}")
        return None


def extraer_parametros_del_nombre(ruta_volumen, nombre_imagen):
    # Extrae la parte del nombre del archivo entre <imagen>_ y .nii.gz
    # que contiene los parametros codificados.
    stem = ruta_volumen.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    # Quitar prefijo "<imagen>_" si esta presente
    prefijo = nombre_imagen + "_"
    if stem.startswith(prefijo):
        stem = stem[len(prefijo):]
    return stem


def listar_volumenes(carpeta, nombre_imagen):
    # Lista los .nii.gz que empiezan con <imagen>_ dentro de la carpeta dada.
    if not carpeta.exists():
        return []
    archivos = sorted(carpeta.glob(f"{nombre_imagen}_*.nii.gz"))
    return archivos


def cortes_centrales(volumen):
    # Devuelve los 3 cortes centrales (axial, coronal, sagital) ya orientados.
    nz, ny, nx = volumen.shape
    axial = np.rot90(volumen[nz // 2, :, :])
    coronal = np.rot90(volumen[:, ny // 2, :])
    sagital = volumen[:, :, nx // 2]
    return axial, coronal, sagital


# ---------------------------------------------------------------------------
# Construccion del mosaico
# ---------------------------------------------------------------------------
def dibujar_celda_watershed(gs_celda, fig, volumen, titulo_celda):
    # 3 subplots horizontales sin separacion - label map en nipy_spectral
    sub_gs = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_celda, wspace=0, hspace=0
    )
    axial, coronal, sagital = cortes_centrales(volumen)
    cortes = [axial, coronal, sagital]

    # Titulo de la celda en el primer subplot (centrado encima del grupo)
    for j, corte in enumerate(cortes):
        ax = fig.add_subplot(sub_gs[0, j])
        ax.imshow(corte, cmap="nipy_spectral", aspect="auto")
        ax.axis("off")
        if j == 1:
            # Subtitulo centrado encima de la celda (eje del medio)
            ax.set_title(titulo_celda, fontsize=6)


def dibujar_celda_level_sets(gs_celda, fig, volumen_original, volumen_mascara,
                              titulo_celda):
    # 3 subplots horizontales: imagen original en gris + contorno rojo
    sub_gs = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_celda, wspace=0, hspace=0
    )
    orig = cortes_centrales(volumen_original)
    masc = cortes_centrales(volumen_mascara)
    vmax = float(np.percentile(volumen_original, 98))

    for j in range(3):
        ax = fig.add_subplot(sub_gs[0, j])
        ax.imshow(orig[j], cmap="gray", vmin=0, vmax=vmax, aspect="auto")
        if np.any(masc[j] > 0):
            ax.contour(masc[j] > 0, levels=[0.5], colors="red", linewidths=0.8)
        ax.axis("off")
        if j == 1:
            ax.set_title(titulo_celda, fontsize=6)


def generar_mosaico(nombre_imagen, nombre_metodo, archivos_volumenes):
    # Construye el mosaico completo y lo guarda. Devuelve True si se genero.
    n_celdas = len(archivos_volumenes)
    n_cols = min(MAX_CELDAS_POR_FILA, n_celdas)
    n_filas = math.ceil(n_celdas / MAX_CELDAS_POR_FILA)

    # Cada celda ocupa el ancho de 3 subplots, asi que damos un poco de
    # espacio horizontal extra para las 3 vistas internas.
    fig = plt.figure(
        figsize=(4.5 * n_cols, 2.2 * n_filas + 0.6),
        facecolor="white",
    )
    # Hueco superior reservado para el suptitle/subtitle
    gs = GridSpec(
        n_filas, n_cols, figure=fig,
        wspace=0.05, hspace=0.25,
        top=0.88, bottom=0.02, left=0.02, right=0.98,
    )

    # Para level_sets cargamos la imagen original del paciente una sola vez
    volumen_original = None
    if nombre_metodo.startswith("level_sets"):
        ruta_original = IMAGES_DIR / f"{nombre_imagen}.nii.gz"
        if ruta_original.exists():
            volumen_original = leer_volumen(ruta_original)
        if volumen_original is None:
            print(f"  [aviso] no se encontro imagen original {ruta_original}; "
                  f"el contorno rojo necesita la anatomia de fondo")

    celdas_dibujadas = 0
    for idx, ruta_vol in enumerate(archivos_volumenes):
        fila = idx // MAX_CELDAS_POR_FILA
        columna = idx % MAX_CELDAS_POR_FILA

        volumen = leer_volumen(ruta_vol)
        if volumen is None:
            continue

        titulo_celda = extraer_parametros_del_nombre(ruta_vol, nombre_imagen)

        gs_celda = gs[fila, columna]
        if nombre_metodo == "watershed":
            dibujar_celda_watershed(gs_celda, fig, volumen, titulo_celda)
        else:
            if volumen_original is None:
                # Sin imagen original; mostramos solo la mascara en gris para no romper
                dibujar_celda_watershed(gs_celda, fig, volumen, titulo_celda)
            else:
                dibujar_celda_level_sets(
                    gs_celda, fig, volumen_original, volumen, titulo_celda
                )
        celdas_dibujadas += 1

    if celdas_dibujadas == 0:
        plt.close(fig)
        return False

    # Titulo principal y subtitulo (parametros fijos)
    titulo_principal = (
        f"{nombre_imagen} — {nombre_metodo} — "
        f"{celdas_dibujadas} combinaciones"
    )
    fig.suptitle(titulo_principal, fontsize=10, fontweight="bold")

    sub = SUBTITULOS_FIJOS.get(nombre_metodo)
    if sub:
        fig.text(0.5, 0.93, sub, ha="center", va="top", fontsize=8)

    # Guardado
    os.makedirs(OUTPUT_MOSAICS_DIR, exist_ok=True)
    ruta_salida = OUTPUT_MOSAICS_DIR / f"{nombre_imagen}_{nombre_metodo}_mosaic.png"
    fig.savefig(str(ruta_salida), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Mosaico guardado: {ruta_salida}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_MOSAICS_DIR, exist_ok=True)

    # Inventario de combinaciones disponibles
    print("=" * 70)
    print("Generacion automatica de mosaicos")
    print("=" * 70)
    print("Imagenes encontradas:")

    inventario = []  # lista de (imagen, metodo, lista_archivos)
    for imagen in IMAGENES:
        for nombre_metodo, carpeta_str in METODOS.items():
            carpeta = Path(carpeta_str)
            archivos = listar_volumenes(carpeta, imagen)
            if archivos:
                print(f"  {imagen:14s} — {nombre_metodo:18s}: "
                      f"{len(archivos):3d} volumenes")
                inventario.append((imagen, nombre_metodo, archivos))
            else:
                print(f"  {imagen:14s} — {nombre_metodo:18s}: "
                      f"  0 volumenes (sin resultados, se omite)")

    print("-" * 70)
    total_a_procesar = len(inventario)
    if total_a_procesar == 0:
        print("No hay resultados disponibles para generar mosaicos.")
        return

    # Generar un mosaico por combinacion (imagen, metodo)
    generados = 0
    for imagen, nombre_metodo, archivos in inventario:
        print(f"\nProcesando: {imagen} - {nombre_metodo} "
              f"({len(archivos)} volumenes)")
        if generar_mosaico(imagen, nombre_metodo, archivos):
            generados += 1

    print("\n" + "=" * 70)
    print(f"Mosaicos generados: {generados} de {total_a_procesar} combinaciones procesadas")
    print("=" * 70)


if __name__ == "__main__":
    main()
