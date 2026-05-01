import itk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Parametros generales
Dimension = 3
PixelType = itk.US
ImageType = itk.Image[PixelType, Dimension]

# Coordenadas de la semilla (volumen 288 x 320 x 208 = X x Y x Z)
SEED_X = 132  # columna (eje X del array numpy: axis=2)
SEED_Y = 142  # fila    (eje Y del array numpy: axis=1)
SEED_Z = 96   # slice   (eje Z del array numpy: axis=0)

# Definicion de los tres planos ortogonales que pasan por la semilla.
# Por la orientacion del NIfTI A1_grayT1, el eje del array que corresponde
# a cada vista anatomica esta "rotado" respecto al orden ITK (Z, Y, X):
# (titulo, indice de eje en numpy, indice del corte)
PLANOS = [
    ("Axial (z=96)",    2, SEED_X),   # corte sobre eje X del array -> vista superior
    ("Coronal (y=142)", 1, SEED_Y),   # corte sobre eje Y del array -> vista frontal
    ("Sagital (x=132)", 0, SEED_Z),   # corte sobre eje Z del array -> vista lateral
]

# Archivos de segmentacion y etiquetas por fila
ARCHIVOS = [
    ("connected_lower100_upper170.nii.gz", "Lower=100 Upper=170 (estrecha)"),
    ("connected_lower80_upper200.nii.gz",  "Lower=80  Upper=200 (media)"),
    ("connected_lower50_upper250.nii.gz",  "Lower=50  Upper=250 (amplia)"),
]


def extraer_corte(volumen, eje, indice):
    # Devuelve el corte 2D ya orientado anatomicamente
    # (cabeza arriba, anterior al frente).
    if eje == 2:
        # Axial: vista superior, requiere rotacion 90 grados
        return np.rot90(volumen[:, :, indice])
    if eje == 1:
        # Coronal: vista frontal, requiere rotacion 90 grados
        return np.rot90(volumen[:, indice, :])
    # Sagital: vista lateral, ya queda correcta
    return volumen[indice, :, :]


def leer_imagen(ruta):
    # Lee una imagen .nii.gz como itk.US y la convierte a numpy
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(ruta)
    reader.Update()
    return itk.array_view_from_image(reader.GetOutput())


# Cargar imagen original
original = leer_imagen("A1_grayT1.nii.gz")
vmax = float(np.percentile(original, 98))

# Cargar las 3 segmentaciones
segmentaciones = [(etiqueta, leer_imagen(ruta)) for ruta, etiqueta in ARCHIVOS]

# Mapa de color para superponer la mascara en cian (transparente fuera de la region)
cmap_overlay = ListedColormap([(0, 0, 0, 0), (0.0, 1.0, 1.0, 0.55)])

N_FILAS = len(ARCHIVOS)
fig, axes = plt.subplots(N_FILAS, 3, figsize=(14, 5 * N_FILAS),
                         constrained_layout=True)

for i, (etiqueta, mascara) in enumerate(segmentaciones):
    total_voxeles = int(np.count_nonzero(mascara))
    for j, (titulo_plano, eje, indice) in enumerate(PLANOS):
        ax = axes[i, j]
        corte_original = extraer_corte(original, eje, indice)
        corte_mascara  = extraer_corte(mascara,  eje, indice)
        # Imagen original en escala de grises
        ax.imshow(corte_original, cmap="gray", vmin=0, vmax=vmax,
                  aspect="auto")
        # Superponer mascara binarizada en cian
        ax.imshow((corte_mascara > 0).astype(np.uint8),
                  cmap=cmap_overlay, vmin=0, vmax=1, aspect="auto")
        # Titulo de columna en la fila superior
        if i == 0:
            ax.set_title(titulo_plano, fontsize=11)
        # Etiqueta de fila a la izquierda
        if j == 0:
            ax.set_ylabel(etiqueta, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # Numero de voxeles segmentados en este plano y total
        voxeles_plano = int(np.count_nonzero(corte_mascara))
        ax.text(0.02, 0.98,
                f"vox plano: {voxeles_plano}\nvox total: {total_voxeles}",
                transform=ax.transAxes, fontsize=8, color="yellow",
                ha="left", va="top",
                bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))

fig.suptitle("ConnectedThreshold - Comparación de umbrales", fontsize=14)
fig.savefig("mosaico_connected_threshold.png", dpi=150)
