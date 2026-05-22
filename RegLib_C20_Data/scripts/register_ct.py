"""
Taller 4 — Registro de Imágenes Médicas: RegLib Case #20
Script: register_ct.py

Registro intra-sujeto CT_2 -> CT_1 mediante un pipeline multi-etapa en ITK:
    1) Registro Rígido    (Euler3DTransform)
    2) Registro Affine    (AffineTransform, inicializado con la rígida)
    3) Registro BSpline   (refinamiento deformable)

El producto final es:
    - results/CT_2_registered.nrrd  : CT_2 llevado al espacio de CT_1
    - results/CT_diff.nrrd          : |CT_1 - CT_2_registered|
    - transforms/ct_bspline_transform.tfm : transform BSpline final
      (será reutilizada por register_pet.py para llevar PET_2 al espacio PET_1)

Convenciones del proyecto:
    BASE_DIR       = .../RegLib_C20_Data
    IMAGES_DIR     = BASE_DIR/images
    RESULTS_DIR    = BASE_DIR/results
    TRANSFORMS_DIR = BASE_DIR/transforms
Pixel type ITK: itk.F (float32) en todos los filtros.
"""

import os
import sys
import time

import itk
import numpy as np
import matplotlib
matplotlib.use('Agg')  # sin display, compatible con servidor/Colab
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec

# Marca de tiempo global para medir la duración total del pipeline
tiempo_inicio_total = time.time()

# -----------------------------------------------------------------------------
# Rutas independientes del directorio de trabajo
# -----------------------------------------------------------------------------
BASE_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IMAGES_DIR     = os.path.join(BASE_DIR, "images")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
TRANSFORMS_DIR = os.path.join(BASE_DIR, "transforms")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TRANSFORMS_DIR, exist_ok=True)

FIXED_PATH  = os.path.join(IMAGES_DIR, "CT_1.nrrd")
MOVING_PATH = os.path.join(IMAGES_DIR, "CT_2.nrrd")

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


# =============================================================================
# Etapa 1 — Registro Rígido (Euler3DTransform)
# =============================================================================
def run_rigid_registration(fixed, moving):
    # Nota: usamos VersorRigid3DTransform en lugar de Euler3DTransform porque
    # ITK Python no envuelve CenteredTransformInitializer para Euler3D.
    # Ambas son rígidas en 3D (rotación + traslación); VersorRigid parametriza
    # la rotación con un versor (cuaternión unitario).
    print("\n[Etapa 1] Registro Rígido (VersorRigid3DTransform) ...")

    TransformType = itk.VersorRigid3DTransform[itk.D]
    initial_transform = TransformType.New()

    # Inicialización por momentos (alinea centros de masa)
    InitializerType = itk.CenteredTransformInitializer[
        TransformType, ImageType, ImageType
    ]
    initializer = InitializerType.New()
    initializer.SetTransform(initial_transform)
    initializer.SetFixedImage(fixed)
    initializer.SetMovingImage(moving)
    initializer.MomentsOn()
    initializer.InitializeTransform()

    # Métrica: Mattes MI
    MetricType = itk.MattesMutualInformationImageToImageMetricv4[
        ImageType, ImageType
    ]
    metric = MetricType.New()
    metric.SetNumberOfHistogramBins(50)

    # Optimizador: Regular Step Gradient Descent
    OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
    optimizer = OptimizerType.New()
    optimizer.SetLearningRate(1.0)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)
    optimizer.SetRelaxationFactor(0.5)

    # Registro v4
    RegistrationType = itk.ImageRegistrationMethodv4[ImageType, ImageType]
    registration = RegistrationType.New()
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(initial_transform)
    registration.InPlaceOn()

    # Esquema multi-resolución sencillo (1 nivel)
    registration.SetNumberOfLevels(1)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])

    registration.Update()

    # Con InPlaceOn(), 'initial_transform' es la misma instancia tipada que se
    # optimizó; registration.GetTransform() devolvería la clase base Transform
    # (sin GetCenter/GetMatrix/GetTranslation accesibles desde Python).
    print(f"  Iteraciones: {optimizer.GetCurrentIteration()}")
    print(f"  Métrica final: {optimizer.GetValue():.6f}")
    print("  Etapa 1 completada.")
    return initial_transform


# =============================================================================
# Etapa 2 — Registro Affine (inicializado con la transform rígida)
# =============================================================================
def run_affine_registration(fixed, moving, rigid_transform):
    print("\n[Etapa 2] Registro Affine (AffineTransform) ...")

    AffineType = itk.AffineTransform[itk.D, Dimension]
    affine_transform = AffineType.New()

    # Inicializar el affine con la rotación + traslación del paso rígido:
    # matriz y centro provenientes de la Euler3D y traslación copiada.
    affine_transform.SetCenter(rigid_transform.GetCenter())
    affine_transform.SetMatrix(rigid_transform.GetMatrix())
    affine_transform.SetTranslation(rigid_transform.GetTranslation())

    MetricType = itk.MattesMutualInformationImageToImageMetricv4[
        ImageType, ImageType
    ]
    metric = MetricType.New()
    metric.SetNumberOfHistogramBins(32)

    OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
    optimizer = OptimizerType.New()
    optimizer.SetLearningRate(0.1)
    optimizer.SetMinimumStepLength(0.0001)
    optimizer.SetNumberOfIterations(100)
    optimizer.SetRelaxationFactor(0.5)

    RegistrationType = itk.ImageRegistrationMethodv4[ImageType, ImageType]
    registration = RegistrationType.New()
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(affine_transform)
    registration.InPlaceOn()

    registration.SetNumberOfLevels(3)
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # MetricSamplingStrategy: 0=NONE, 1=REGULAR, 2=RANDOM (enum interno de ITK)
    registration.SetMetricSamplingStrategy(2)
    registration.SetMetricSamplingPercentage(0.10)

    registration.Update()

    print(f"  Iteraciones: {optimizer.GetCurrentIteration()}")
    print(f"  Métrica final: {optimizer.GetValue():.6f}")
    print("  Etapa 2 completada.")
    return affine_transform


# =============================================================================
# Etapa 3 — Registro BSpline (refinamiento deformable)
# =============================================================================
def run_bspline_registration(fixed, moving, affine_transform):
    print("\n[Etapa 3] Registro BSpline (orden 3) ...")

    SplineOrder = 3
    BSplineType = itk.BSplineTransform[itk.D, Dimension, SplineOrder]
    bspline_transform = BSplineType.New()

    # Inicializar la malla BSpline cubriendo el dominio de la imagen fixed.
    InitializerType = itk.BSplineTransformInitializer[BSplineType, ImageType]
    initializer = InitializerType.New()
    initializer.SetTransform(bspline_transform)
    initializer.SetImage(fixed)
    mesh = itk.Size[Dimension]()
    mesh[0] = 5
    mesh[1] = 5
    mesh[2] = 5
    initializer.SetTransformDomainMeshSize(mesh)
    initializer.InitializeTransform()

    MetricType = itk.MattesMutualInformationImageToImageMetricv4[
        ImageType, ImageType
    ]
    metric = MetricType.New()
    metric.SetNumberOfHistogramBins(32)

    # Optimizador: GradientDescent honra bien el budget de iteraciones y el
    # esquema multi-resolución. Mantenemos 20 it por nivel para acotar tiempo.
    OptimizerType = itk.GradientDescentOptimizerv4Template[itk.D]
    optimizer = OptimizerType.New()
    optimizer.SetLearningRate(1.0)
    optimizer.SetNumberOfIterations(20)
    optimizer.SetConvergenceWindowSize(5)

    RegistrationType = itk.ImageRegistrationMethodv4[ImageType, ImageType]
    registration = RegistrationType.New()
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(bspline_transform)
    registration.InPlaceOn()

    # La transform affine se aplica como "moving initial transform":
    # primero se compone con el BSpline durante la optimización.
    registration.SetMovingInitialTransform(affine_transform)

    # Pirámide multi-resolución: 3 niveles, de grueso a fino.
    # Acelera enormemente respecto a evaluar siempre a full resolution.
    registration.SetNumberOfLevels(3)
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Muestreo estocástico de la métrica MI: ~10% de voxels por iteración.
    # Es la optimización que más acorta el tiempo manteniendo calidad.
    # MetricSamplingStrategy: 0=NONE, 1=REGULAR, 2=RANDOM.
    registration.SetMetricSamplingStrategy(2)
    registration.SetMetricSamplingPercentage(0.10)

    registration.Update()

    print("  Etapa 3 completada.")
    return bspline_transform


# =============================================================================
# Aplicación de la transform final (Affine ∘ BSpline) sobre CT_2
# =============================================================================
def resample_moving(fixed, moving, affine_transform, bspline_transform):
    """Genera la imagen CT_2 registrada en el espacio de CT_1."""
    # Composición: primero BSpline, luego Affine (orden de aplicación a un
    # punto: T_total(p) = T_affine(T_bspline(p))).
    CompositeType = itk.CompositeTransform[itk.D, Dimension]
    composite = CompositeType.New()
    composite.AddTransform(affine_transform)
    composite.AddTransform(bspline_transform)

    ResamplerType = itk.ResampleImageFilter[ImageType, ImageType]
    resampler = ResamplerType.New()
    resampler.SetInput(moving)
    resampler.SetTransform(composite)
    resampler.SetUseReferenceImage(True)
    resampler.SetReferenceImage(fixed)
    resampler.SetDefaultPixelValue(-1000.0)  # HU del aire

    InterpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    interpolator = InterpolatorType.New()
    resampler.SetInterpolator(interpolator)

    resampler.Update()
    return resampler.GetOutput()


def absolute_difference(image_a, image_b):
    """Devuelve |image_a - image_b| como itk.Image[F, 3]."""
    SubFilter = itk.SubtractImageFilter[ImageType, ImageType, ImageType].New()
    SubFilter.SetInput1(image_a)
    SubFilter.SetInput2(image_b)

    AbsFilter = itk.AbsImageFilter[ImageType, ImageType].New()
    AbsFilter.SetInput(SubFilter.GetOutput())
    AbsFilter.Update()
    return AbsFilter.GetOutput()


def save_three_views(image_np, title, output_path, cmap='gray', vmin=None, vmax=None):
    """
    Guarda axial, coronal y sagital del slice central de un volumen numpy.
    image_np: array (Z, Y, X) — resultado de itk.array_from_image()
    """
    z, y, x = image_np.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    vistas = [
        (image_np[z // 2, :, :], "Axial"),
        (image_np[:, y // 2, :], "Coronal"),
        (image_np[:, :, x // 2], "Sagital"),
    ]
    for ax, (slice_2d, nombre) in zip(axes, vistas):
        ax.imshow(slice_2d, cmap=cmap, origin='lower', aspect='auto',
                  vmin=vmin, vmax=vmax)
        ax.set_title(nombre)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Programa principal
# =============================================================================
def main():
    print("=" * 70)
    print("Taller 4 - Registro CT_2 -> CT_1 (RegLib Case #20)")
    print("=" * 70)

    print("\n[1/4] Lectura de imágenes ...")
    fixed  = read_image(FIXED_PATH)
    moving = read_image(MOVING_PATH)
    print_image_info("CT_1 (fixed)",  fixed)
    print_image_info("CT_2 (moving)", moving)

    # --- Pipeline de registro -------------------------------------------------
    tiempo_inicio_etapa = time.time()
    rigid_tx = run_rigid_registration(fixed, moving)
    tiempo_etapa = time.time() - tiempo_inicio_etapa
    print(f"[Etapa 1] Rígido completada ✓ — Tiempo: {tiempo_etapa:.1f}s ({tiempo_etapa/60:.1f} min)")

    tiempo_inicio_etapa = time.time()
    affine_tx = run_affine_registration(fixed, moving, rigid_tx)
    tiempo_etapa = time.time() - tiempo_inicio_etapa
    print(f"[Etapa 2] Affine completada ✓ — Tiempo: {tiempo_etapa:.1f}s ({tiempo_etapa/60:.1f} min)")

    tiempo_inicio_etapa = time.time()
    bspline_tx = run_bspline_registration(fixed, moving, affine_tx)
    tiempo_etapa = time.time() - tiempo_inicio_etapa
    print(f"[Etapa 3] BSpline completada ✓ — Tiempo: {tiempo_etapa:.1f}s ({tiempo_etapa/60:.1f} min)")

    # --- Resampling, diferencia y guardado -----------------------------------
    tiempo_inicio_etapa = time.time()
    print("\n[2/4] Resampling de CT_2 al espacio de CT_1 ...")
    moving_resampled = resample_moving(fixed, moving, affine_tx, bspline_tx)

    print("[3/4] Cálculo de imagen diferencia |CT_1 - CT_2_registered| ...")
    diff_image = absolute_difference(fixed, moving_resampled)

    # --- Guardado de resultados ----------------------------------------------
    print("[4/4] Guardando resultados ...")
    out_registered = os.path.join(RESULTS_DIR, "CT_2_registered.nrrd")
    out_diff       = os.path.join(RESULTS_DIR, "CT_diff.nrrd")
    out_bspline    = os.path.join(TRANSFORMS_DIR, "ct_bspline_transform.tfm")
    out_affine     = os.path.join(TRANSFORMS_DIR, "ct_affine_transform.tfm")

    write_image(moving_resampled, out_registered)
    write_image(diff_image,       out_diff)

    # Guardar la transformación affine y la BSpline por separado;
    # register_pet.py necesitará ambas para componer T_total.
    tx_writer = itk.TransformFileWriterTemplate[itk.D].New()
    tx_writer.SetInput(affine_tx)
    tx_writer.SetFileName(out_affine)
    tx_writer.Update()

    tx_writer = itk.TransformFileWriterTemplate[itk.D].New()
    tx_writer.SetInput(bspline_tx)
    tx_writer.SetFileName(out_bspline)
    tx_writer.Update()

    print("\nArchivos generados:")
    print(f"  - {out_registered}")
    print(f"  - {out_diff}")
    print(f"  - {out_affine}")
    print(f"  - {out_bspline}")

    tiempo_etapa = time.time() - tiempo_inicio_etapa
    print(f"[Etapa 4] Resampling CT y guardado — Tiempo: {tiempo_etapa:.1f}s ({tiempo_etapa/60:.1f} min)")

    print("\nRegistro CT finalizado correctamente.")

    # -------------------------------------------------------------------------
    # Exportación PNG — 3 vistas de cada resultado CT
    # -------------------------------------------------------------------------
    print("\n[5/4] Generando visualizaciones PNG ...")
    tiempo_inicio_etapa = time.time()

    # Convertir imágenes a numpy (eje Z primero)
    ct1_np  = itk.array_from_image(fixed)
    ct2_np  = itk.array_from_image(moving)
    ct2r_np = itk.array_from_image(moving_resampled)
    diff_np = itk.array_from_image(diff_image)

    # Rango de visualización CT: ventana HU estándar tejido blando
    vmin_ct, vmax_ct = -200, 400

    save_three_views(
        ct1_np, "CT_1 (Fixed)",
        os.path.join(RESULTS_DIR, "CT_1_views.png"),
        cmap='gray', vmin=vmin_ct, vmax=vmax_ct
    )
    save_three_views(
        ct2_np, "CT_2 (Moving — sin registrar)",
        os.path.join(RESULTS_DIR, "CT_2_original_views.png"),
        cmap='gray', vmin=vmin_ct, vmax=vmax_ct
    )
    save_three_views(
        ct2r_np, "CT_2 Registrado",
        os.path.join(RESULTS_DIR, "CT_2_registered_views.png"),
        cmap='gray', vmin=vmin_ct, vmax=vmax_ct
    )
    save_three_views(
        diff_np, "Diferencia |CT_1 - CT_2_registered|",
        os.path.join(RESULTS_DIR, "CT_diff_views.png"),
        cmap='hot', vmin=0, vmax=200
    )

    tiempo_etapa = time.time() - tiempo_inicio_etapa
    print(f"[Etapa 5] Visualizaciones PNG — Tiempo: {tiempo_etapa:.1f}s ({tiempo_etapa/60:.1f} min)")
    print(f"  - {RESULTS_DIR}/CT_1_views.png")
    print(f"  - {RESULTS_DIR}/CT_2_original_views.png")
    print(f"  - {RESULTS_DIR}/CT_2_registered_views.png")
    print(f"  - {RESULTS_DIR}/CT_diff_views.png")

    tiempo_total = time.time() - tiempo_inicio_total
    print(f"\n=== Pipeline CT completado ===")
    print(f"Tiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")


if __name__ == "__main__":
    main()
