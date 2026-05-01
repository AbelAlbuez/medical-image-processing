import itk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Visualizacion de resultados: 3 cortes ortogonales (axial, coronal, sagital)
# de un volumen .nii.gz (segmentacion o imagen original).

# ---------------------------------------------------------------------------
# Variables configurables
# ---------------------------------------------------------------------------
IMAGE_PATH = "output/watershed/"  # ruta al .nii.gz a visualizar

# Indices de corte. Si se dejan en None, se usa el centro del volumen en cada eje.
SLICE_AXIAL   = None  # corte sobre eje Z (numpy axis=0)
SLICE_CORONAL = None  # corte sobre eje Y (numpy axis=1)
SLICE_SAGITAL = None  # corte sobre eje X (numpy axis=2)

# True  -> volumen de etiquetas (colormap 'nipy_spectral'), ideal para
#          watersheds y otras segmentaciones con muchas regiones.
# False -> volumen original en escala de grises ('gray').
USE_LABEL_COLORMAP = True

# Ruta de salida de la figura. Si None, se guarda junto al .nii.gz con sufijo _vista.png
OUTPUT_FIGURE = None
# ---------------------------------------------------------------------------

# Parametros generales
Dimension = 3


def leer_volumen(ruta):
    # Lee la imagen como float (compatible con label maps y volumenes originales)
    # y devuelve el array numpy en orden (Z, Y, X).
    imagen = itk.imread(str(ruta), itk.F)
    return itk.array_view_from_image(imagen)


def extraer_corte(volumen, eje, indice):
    # Devuelve el corte 2D ya orientado anatomicamente
    # (cabeza arriba, anterior al frente).
    if eje == 0:
        # Axial: vista superior, requiere rotacion 90 grados
        return np.rot90(volumen[indice, :, :])
    if eje == 1:
        # Coronal: vista frontal, requiere rotacion 90 grados
        return np.rot90(volumen[:, indice, :])
    # Sagital: vista lateral
    return volumen[:, :, indice]


def main():
    ruta = Path(IMAGE_PATH)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro la imagen: {ruta}")

    volumen = leer_volumen(ruta)
    nz, ny, nx = volumen.shape

    # Resolver indices de corte (centros del volumen si quedan en None)
    idx_axial   = SLICE_AXIAL   if SLICE_AXIAL   is not None else nz // 2
    idx_coronal = SLICE_CORONAL if SLICE_CORONAL is not None else ny // 2
    idx_sagital = SLICE_SAGITAL if SLICE_SAGITAL is not None else nx // 2

    PLANOS = [
        (f"Axial (z={idx_axial})",     0, idx_axial),
        (f"Coronal (y={idx_coronal})", 1, idx_coronal),
        (f"Sagital (x={idx_sagital})", 2, idx_sagital),
    ]

    # Seleccion de colormap segun el tipo de volumen
    if USE_LABEL_COLORMAP:
        # nipy_spectral asigna colores distintos a etiquetas consecutivas;
        # imprescindible para distinguir cuencas adyacentes en watershed.
        cmap = "nipy_spectral"
        vmin, vmax = float(volumen.min()), float(volumen.max())
    else:
        cmap = "gray"
        vmin = 0.0
        vmax = float(np.percentile(volumen, 98))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, (titulo, eje, indice) in zip(axes, PLANOS):
        corte = extraer_corte(volumen, eje, indice)
        ax.imshow(corte, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(titulo, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(ruta.name, fontsize=12)

    # Resolver ruta de salida
    if OUTPUT_FIGURE is None:
        ruta_figura = ruta.with_name(ruta.stem.replace(".nii", "") + "_vista.png")
    else:
        ruta_figura = Path(OUTPUT_FIGURE)
    fig.savefig(str(ruta_figura), dpi=150)
    plt.close(fig)
    print(f"Figura guardada en: {ruta_figura}")


if __name__ == "__main__":
    main()
