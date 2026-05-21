"""
Taller 4 — Registro de Imágenes Médicas: RegLib Case #20
Script: register_pet.py

Aplica al volumen PET_2 la misma transformación geométrica compuesta
(affine ∘ BSpline) estimada por register_ct.py sobre el par CT.
Dado que PET_2 fue adquirido co-registrado con CT_2 y PET_1 con CT_1, esa
misma transformación lleva PET_2 al espacio de PET_1 sin necesidad de
volver a optimizar.

Productos:
    - results/PET_2_registered.nrrd
    - results/PET_diff.nrrd                (|PET_1 - PET_2_registered|)

Prerrequisito: haber ejecutado previamente scripts/register_ct.py, de modo
que transforms/ct_affine_transform.tfm y transforms/ct_bspline_transform.tfm
existan en disco.
"""

import os
import sys
import itk

# -----------------------------------------------------------------------------
# Rutas independientes del directorio de trabajo
# -----------------------------------------------------------------------------
BASE_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IMAGES_DIR     = os.path.join(BASE_DIR, "images")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
TRANSFORMS_DIR = os.path.join(BASE_DIR, "transforms")

os.makedirs(RESULTS_DIR, exist_ok=True)

FIXED_PATH   = os.path.join(IMAGES_DIR, "PET_1.nrrd")
MOVING_PATH  = os.path.join(IMAGES_DIR, "PET_2.nrrd")
AFFINE_PATH  = os.path.join(TRANSFORMS_DIR, "ct_affine_transform.tfm")
BSPLINE_PATH = os.path.join(TRANSFORMS_DIR, "ct_bspline_transform.tfm")

# -----------------------------------------------------------------------------
# Tipos ITK
# -----------------------------------------------------------------------------
PixelType = itk.F
Dimension = 3
ImageType = itk.Image[PixelType, Dimension]


# =============================================================================
# Utilidades de E/S
# =============================================================================
def read_image(path):
    """Lee una imagen NRRD y la devuelve como itk.Image[F, 3]."""
    try:
        reader = itk.ImageFileReader[ImageType].New()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()
    except Exception as exc:
        print(f"[ERROR] No fue posible leer la imagen: {path}")
        print(f"        Detalle: {exc}")
        sys.exit(1)


def write_image(image, path):
    """Escribe una itk.Image[F, 3] en disco."""
    writer = itk.ImageFileWriter[ImageType].New()
    writer.SetFileName(path)
    writer.SetInput(image)
    writer.Update()


def print_image_info(name, image):
    """Imprime metadata principal de una imagen ITK."""
    size    = image.GetLargestPossibleRegion().GetSize()
    spacing = image.GetSpacing()
    origin  = image.GetOrigin()
    print(f"  {name}:")
    print(f"    size    = [{size[0]}, {size[1]}, {size[2]}]")
    print(f"    spacing = [{spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f}]")
    print(f"    origin  = [{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}]")


def read_transform(path):
    """Lee una transformación ITK desde un archivo .tfm.

    itk.transformread devuelve una lista (puede contener varias transformadas
    si el archivo agrega varias); aquí esperamos siempre la primera.
    """
    try:
        transforms = itk.transformread(path)
        if not transforms:
            raise RuntimeError("El archivo no contiene transformaciones.")
        return transforms[0]
    except Exception as exc:
        print(f"[ERROR] No fue posible leer la transformación: {path}")
        print(f"        Detalle: {exc}")
        sys.exit(1)


# =============================================================================
# Aplicación de la transformación compuesta a PET_2
# =============================================================================
def resample_pet(fixed_ref, moving, affine_tx, bspline_tx):
    """Aplica (affine ∘ BSpline) sobre 'moving' usando 'fixed_ref' como rejilla."""
    # CompositeTransform: mismo orden de inserción usado en register_ct.py
    # (primero la affine, luego la BSpline).
    CompositeType = itk.CompositeTransform[itk.D, Dimension]
    composite = CompositeType.New()
    composite.AddTransform(affine_tx)
    composite.AddTransform(bspline_tx)

    ResamplerType = itk.ResampleImageFilter[ImageType, ImageType]
    resampler = ResamplerType.New()
    resampler.SetInput(moving)
    resampler.SetTransform(composite)
    resampler.SetReferenceImage(fixed_ref)
    resampler.UseReferenceImageOn()
    resampler.SetDefaultPixelValue(0.0)  # SUV mínimo apropiado para PET

    InterpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    resampler.SetInterpolator(InterpolatorType.New())

    resampler.Update()
    return resampler.GetOutput()


def absolute_difference(image_a, image_b):
    """Devuelve |image_a - image_b| como itk.Image[F, 3]."""
    sub = itk.SubtractImageFilter[ImageType, ImageType, ImageType].New()
    sub.SetInput1(image_a)
    sub.SetInput2(image_b)

    abs_filter = itk.AbsImageFilter[ImageType, ImageType].New()
    abs_filter.SetInput(sub.GetOutput())
    abs_filter.Update()
    return abs_filter.GetOutput()


# =============================================================================
# Programa principal
# =============================================================================
def main():
    print("=" * 70)
    print("Taller 4 - Transferencia de transform CT al par PET (RegLib C20)")
    print("=" * 70)

    # --- Lectura de imágenes PET --------------------------------------------
    print("\n[1/4] Lectura de imágenes PET ...")
    pet_fixed  = read_image(FIXED_PATH)
    pet_moving = read_image(MOVING_PATH)
    print_image_info("PET_1 (referencia)", pet_fixed)
    print_image_info("PET_2 (moving)",     pet_moving)

    # --- Lectura de transformaciones CT -------------------------------------
    print("\n[2/4] Leyendo transformaciones CT ...")
    if not os.path.isfile(AFFINE_PATH) or not os.path.isfile(BSPLINE_PATH):
        print("[ERROR] No se encontraron las transformaciones CT en:")
        print(f"          {AFFINE_PATH}")
        print(f"          {BSPLINE_PATH}")
        print("        Ejecuta primero: python scripts/register_ct.py")
        sys.exit(1)

    affine_tx  = read_transform(AFFINE_PATH)
    bspline_tx = read_transform(BSPLINE_PATH)
    print(f"  Affine  : {type(affine_tx).__name__}")
    print(f"  BSpline : {type(bspline_tx).__name__}")

    # --- Resampling de PET_2 ------------------------------------------------
    print("\n[3/4] Aplicando transformación a PET_2 ...")
    pet_registered = resample_pet(pet_fixed, pet_moving, affine_tx, bspline_tx)

    print("       Calculando imagen diferencia |PET_1 - PET_2_registered| ...")
    pet_diff = absolute_difference(pet_fixed, pet_registered)

    # --- Guardado de resultados --------------------------------------------
    print("\n[4/4] Guardando resultados ...")
    out_registered = os.path.join(RESULTS_DIR, "PET_2_registered.nrrd")
    out_diff       = os.path.join(RESULTS_DIR, "PET_diff.nrrd")

    write_image(pet_registered, out_registered)
    write_image(pet_diff,       out_diff)

    print("\nArchivos generados:")
    print(f"  - {out_registered}")
    print(f"  - {out_diff}")
    print("\nRegistro PET finalizado correctamente.")


if __name__ == "__main__":
    main()
