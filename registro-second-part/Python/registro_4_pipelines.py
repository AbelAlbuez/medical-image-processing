import os
import math
from pathlib import Path

import itk
import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
IMG_DIR = ROOT / "imgs"
OUT_ROOT = ROOT / "output"


def save_mosaic(arrays, title, out_dir, params_text=None):
    arr_fixed, arr_moving, arr_registered, arr_diff_before, arr_diff_after = arrays
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    # Cada diff usa su propia escala simétrica (percentil 99 → robusto a outliers
    # de los píxeles de fondo introducidos por DefaultPixelValue).
    def _sym_range(a):
        v = float(np.percentile(np.abs(a), 99))
        return v if v > 1e-9 else 1e-9

    rng_before = _sym_range(arr_diff_before)
    rng_after = _sym_range(arr_diff_after)

    panels = [
        (axes[0, 0], arr_fixed,       "Fija",        "gray",   None),
        (axes[0, 1], arr_moving,      "Movible",     "gray",   None),
        (axes[0, 2], arr_registered,  "Registrada",  "gray",   None),
        (axes[1, 0], arr_diff_before, f"Diferencia ANTES (±{rng_before:.2g})",   "RdBu_r", rng_before),
        (axes[1, 1], arr_diff_after,  f"Diferencia DESPUÉS (±{rng_after:.2g})",  "RdBu_r", rng_after),
    ]
    for ax, arr, label, cmap, rng in panels:
        if rng is not None:
            im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=-rng, vmax=rng)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(label)
        ax.axis("off")
    axes[1, 2].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    if params_text:
        fig.text(
            0.5, 0.02, params_text,
            ha="center", va="bottom",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#888"),
        )
    nombre = out_dir.split("/")[-1]
    plt.savefig(f"{out_dir}/mosaico_{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def guardar_individuales(out_dir, arr_fixed, arr_moving, arr_registered, arr_diff):
    plt.imsave(f"{out_dir}/fixed.png", arr_fixed, cmap="gray", origin="lower")
    plt.imsave(f"{out_dir}/moving.png", arr_moving, cmap="gray", origin="lower")
    plt.imsave(f"{out_dir}/registered.png", arr_registered, cmap="gray", origin="lower")
    plt.imsave(f"{out_dir}/difference.png", arr_diff, cmap="RdBu_r", origin="lower")


def attach_iteration_logger(optimizer, prefix):
    def _cb():
        it = optimizer.GetCurrentIteration()
        val = optimizer.GetValue()
        pos = optimizer.GetCurrentPosition()
        params = [pos[i] for i in range(len(pos))]
        print(f"  [{prefix}] iter={it} metric={val:.6f} params={params}")
    cmd = itk.PyCommand.New()
    cmd.SetCommandCallable(_cb)
    optimizer.AddObserver(itk.IterationEvent(), cmd)


def pipeline1_traslacion2D_monomodal():
    print("\n=== Pipeline 1: Traslación 2D (monomodal, Mean Squares) ===")
    out_dir = str(OUT_ROOT / "pipeline1_traslacion2D_monomodal")
    os.makedirs(out_dir, exist_ok=True)

    fixed_path = str(IMG_DIR / "BrainProtonDensitySliceBorder20.png")
    moving_path = str(IMG_DIR / "BrainProtonDensitySliceShifted13x17y.png")

    Dimension = 2
    PixelType = itk.F
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    TransformType = itk.TranslationTransform[itk.D, Dimension]
    OptimizerType = itk.RegularStepGradientDescentOptimizer
    MetricType = itk.MeanSquaresImageToImageMetric[FixedImageType, MovingImageType]
    InterpolatorType = itk.LinearInterpolateImageFunction[MovingImageType, itk.D]
    RegistrationType = itk.ImageRegistrationMethod[FixedImageType, MovingImageType]

    metric = MetricType.New()
    transform = TransformType.New()
    optimizer = OptimizerType.New()
    interpolator = InterpolatorType.New()
    registration = RegistrationType.New()

    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetTransform(transform)
    registration.SetInterpolator(interpolator)

    fixedReader = itk.ImageFileReader[FixedImageType].New()
    movingReader = itk.ImageFileReader[MovingImageType].New()
    fixedReader.SetFileName(fixed_path)
    movingReader.SetFileName(moving_path)
    fixedReader.Update()
    movingReader.Update()

    registration.SetFixedImage(fixedReader.GetOutput())
    registration.SetMovingImage(movingReader.GetOutput())
    registration.SetFixedImageRegion(fixedReader.GetOutput().GetBufferedRegion())

    initialParameters = itk.OptimizerParameters[itk.D](transform.GetNumberOfParameters())
    initialParameters[0] = 0.0
    initialParameters[1] = 0.0
    registration.SetInitialTransformParameters(initialParameters)

    optimizer.SetMaximumStepLength(4.00)
    optimizer.SetMinimumStepLength(0.01)
    optimizer.SetNumberOfIterations(200)
    attach_iteration_logger(optimizer, "p1")

    registration.Update()

    finalParameters = registration.GetLastTransformParameters()
    print(f"  Translation X = {finalParameters[0]}")
    print(f"  Translation Y = {finalParameters[1]}")
    print(f"  Iterations    = {optimizer.GetCurrentIteration()}")
    print(f"  Metric value  = {optimizer.GetValue()}")

    fixedImage = fixedReader.GetOutput()
    resampler = itk.ResampleImageFilter[MovingImageType, FixedImageType].New()
    resampler.SetInput(movingReader.GetOutput())
    resampler.SetTransform(registration.GetOutput().Get())
    resampler.SetSize(fixedImage.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixedImage.GetOrigin())
    resampler.SetOutputSpacing(fixedImage.GetSpacing())
    resampler.SetOutputDirection(fixedImage.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    arr_fixed = itk.array_from_image(fixedReader.GetOutput())
    arr_moving = itk.array_from_image(movingReader.GetOutput())
    arr_registered = itk.array_from_image(resampler.GetOutput())
    arr_diff_before = arr_fixed - arr_moving
    arr_diff = arr_fixed - arr_registered

    guardar_individuales(out_dir, arr_fixed, arr_moving, arr_registered, arr_diff)
    params_text = (
        "Transformación: TranslationTransform 2D   |   Métrica: MeanSquares   |   "
        "Optimizer: RegularStepGradientDescent\n"
        "Config: maxStep=4.00, minStep=0.01, maxIters=200\n"
        f"Resultado: Tx={finalParameters[0]:.4f}, Ty={finalParameters[1]:.4f}   "
        f"iters={optimizer.GetCurrentIteration()}   métrica={optimizer.GetValue():.4f}"
    )
    save_mosaic(
        [arr_fixed, arr_moving, arr_registered, arr_diff_before, arr_diff],
        "Pipeline 1: Traslación 2D (monomodal)",
        out_dir,
        params_text=params_text,
    )


def pipeline2_traslacion2D_multimodal():
    print("\n=== Pipeline 2: Traslación 2D (multimodal, Mutual Information) ===")
    out_dir = str(OUT_ROOT / "pipeline2_traslacion2D_multimodal")
    os.makedirs(out_dir, exist_ok=True)

    fixed_path = str(IMG_DIR / "BrainT1SliceBorder20.png")
    moving_path = str(IMG_DIR / "BrainProtonDensitySliceShifted13x17y.png")

    Dimension = 2
    PixelType = itk.US
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    InternalPixelType = itk.F
    InternalImageType = itk.Image[InternalPixelType, Dimension]

    TransformType = itk.TranslationTransform[itk.D, Dimension]
    OptimizerType = itk.GradientDescentOptimizer
    InterpolatorType = itk.LinearInterpolateImageFunction[InternalImageType, itk.D]
    RegistrationType = itk.ImageRegistrationMethod[InternalImageType, InternalImageType]
    MetricType = itk.MutualInformationImageToImageMetric[InternalImageType, InternalImageType]

    transform = TransformType.New()
    optimizer = OptimizerType.New()
    interpolator = InterpolatorType.New()
    registration = RegistrationType.New()
    metric = MetricType.New()

    registration.SetOptimizer(optimizer)
    registration.SetTransform(transform)
    registration.SetInterpolator(interpolator)
    registration.SetMetric(metric)

    metric.SetFixedImageStandardDeviation(0.4)
    metric.SetMovingImageStandardDeviation(0.4)

    fixedReader = itk.ImageFileReader[FixedImageType].New()
    movingReader = itk.ImageFileReader[MovingImageType].New()
    fixedReader.SetFileName(fixed_path)
    movingReader.SetFileName(moving_path)

    in1caster = itk.CastImageFilter[FixedImageType, InternalImageType].New()
    in1caster.SetInput(fixedReader.GetOutput())
    in2caster = itk.CastImageFilter[MovingImageType, InternalImageType].New()
    in2caster.SetInput(movingReader.GetOutput())

    fixedNormalizer = itk.NormalizeImageFilter[InternalImageType, InternalImageType].New()
    movingNormalizer = itk.NormalizeImageFilter[InternalImageType, InternalImageType].New()
    fixedNormalizer.SetInput(in1caster.GetOutput())
    movingNormalizer.SetInput(in2caster.GetOutput())

    fixedSmoother = itk.DiscreteGaussianImageFilter[InternalImageType, InternalImageType].New()
    movingSmoother = itk.DiscreteGaussianImageFilter[InternalImageType, InternalImageType].New()
    fixedSmoother.SetVariance(2.0)
    movingSmoother.SetVariance(2.0)
    fixedSmoother.SetInput(fixedNormalizer.GetOutput())
    movingSmoother.SetInput(movingNormalizer.GetOutput())

    registration.SetFixedImage(fixedSmoother.GetOutput())
    registration.SetMovingImage(movingSmoother.GetOutput())

    fixedNormalizer.Update()
    fixedImageRegion = fixedNormalizer.GetOutput().GetBufferedRegion()
    registration.SetFixedImageRegion(fixedImageRegion)

    initialParameters = itk.OptimizerParameters[itk.D](transform.GetNumberOfParameters())
    initialParameters[0] = 0.0
    initialParameters[1] = 0.0
    registration.SetInitialTransformParameters(initialParameters)

    numberOfPixels = fixedImageRegion.GetNumberOfPixels()
    numberOfSamples = int(numberOfPixels * 0.01)
    metric.SetNumberOfSpatialSamples(numberOfSamples)

    optimizer.SetLearningRate(15.0)
    optimizer.SetNumberOfIterations(200)
    optimizer.MaximizeOn()
    attach_iteration_logger(optimizer, "p2")

    registration.Update()
    print("  Optimizer stop:", registration.GetOptimizer().GetStopConditionDescription())

    finalParameters = registration.GetLastTransformParameters()
    print(f"  Translation X = {finalParameters[0]}")
    print(f"  Translation Y = {finalParameters[1]}")
    print(f"  Iterations    = {optimizer.GetCurrentIteration()}")
    print(f"  Metric value  = {optimizer.GetValue()}")
    print(f"  Samples       = {numberOfSamples}")

    finalTransform = TransformType.New()
    finalTransform.SetParameters(finalParameters)
    finalTransform.SetFixedParameters(transform.GetFixedParameters())

    fixedImage = fixedReader.GetOutput()
    fixedReader.Update()
    movingReader.Update()

    resampler = itk.ResampleImageFilter[MovingImageType, FixedImageType].New()
    resampler.SetTransform(finalTransform)
    resampler.SetInput(movingReader.GetOutput())
    resampler.SetSize(fixedImage.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixedImage.GetOrigin())
    resampler.SetOutputSpacing(fixedImage.GetSpacing())
    resampler.SetOutputDirection(fixedImage.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    arr_fixed = itk.array_from_image(fixedReader.GetOutput()).astype(np.float32)
    arr_moving = itk.array_from_image(movingReader.GetOutput()).astype(np.float32)
    arr_registered = itk.array_from_image(resampler.GetOutput()).astype(np.float32)

    def _norm(a):
        a = a.astype(np.float32)
        amin, amax = float(a.min()), float(a.max())
        if amax - amin < 1e-9:
            return np.zeros_like(a)
        return (a - amin) / (amax - amin)

    arr_diff_before = _norm(arr_fixed) - _norm(arr_moving)
    arr_diff = _norm(arr_fixed) - _norm(arr_registered)

    guardar_individuales(out_dir, arr_fixed, arr_moving, arr_registered, arr_diff)
    params_text = (
        "Transformación: TranslationTransform 2D   |   Métrica: MutualInformation   |   "
        "Optimizer: GradientDescent (Maximize)\n"
        "Preproc: Cast→Normalize→GaussianSmooth(var=2.0)   "
        f"σ_fija=σ_movible=0.4   muestras={numberOfSamples}\n"
        "Config: learningRate=15.0, maxIters=200\n"
        f"Resultado: Tx={finalParameters[0]:.4f}, Ty={finalParameters[1]:.4f}   "
        f"iters={optimizer.GetCurrentIteration()}   métrica={optimizer.GetValue():.4f}"
    )
    save_mosaic(
        [arr_fixed, arr_moving, arr_registered, arr_diff_before, arr_diff],
        "Pipeline 2: Traslación 2D (multimodal, MI)",
        out_dir,
        params_text=params_text,
    )


def pipeline3_rigido2D():
    print("\n=== Pipeline 3: Rígido 2D (CenteredRigid2DTransform) ===")
    out_dir = str(OUT_ROOT / "pipeline3_rigido2D")
    os.makedirs(out_dir, exist_ok=True)

    fixed_path = str(IMG_DIR / "BrainProtonDensitySliceBorder20.png")
    moving_path = str(IMG_DIR / "BrainProtonDensitySliceR10X13Y17.png")

    Dimension = 2
    PixelType = itk.UC
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    TransformType = itk.CenteredRigid2DTransform[itk.D]
    OptimizerType = itk.RegularStepGradientDescentOptimizer
    MetricType = itk.MeanSquaresImageToImageMetric[FixedImageType, MovingImageType]
    InterpolatorType = itk.LinearInterpolateImageFunction[MovingImageType, itk.D]
    RegistrationType = itk.ImageRegistrationMethod[FixedImageType, MovingImageType]

    metric = MetricType.New()
    optimizer = OptimizerType.New()
    interpolator = InterpolatorType.New()
    registration = RegistrationType.New()
    transform = TransformType.New()

    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInterpolator(interpolator)
    registration.SetTransform(transform)

    fixedReader = itk.ImageFileReader[FixedImageType].New()
    movingReader = itk.ImageFileReader[MovingImageType].New()
    fixedReader.SetFileName(fixed_path)
    movingReader.SetFileName(moving_path)
    fixedReader.Update()
    movingReader.Update()

    registration.SetFixedImage(fixedReader.GetOutput())
    registration.SetMovingImage(movingReader.GetOutput())
    registration.SetFixedImageRegion(fixedReader.GetOutput().GetBufferedRegion())

    fixedImage = fixedReader.GetOutput()
    movingImage = movingReader.GetOutput()

    fSpacing, fOrigin, fSize = fixedImage.GetSpacing(), fixedImage.GetOrigin(), fixedImage.GetLargestPossibleRegion().GetSize()
    cFixed = [fOrigin[0] + fSpacing[0] * fSize[0] / 2.0,
              fOrigin[1] + fSpacing[1] * fSize[1] / 2.0]
    centerFixed = itk.Point[itk.D, 2](cFixed)

    mSpacing, mOrigin, mSize = movingImage.GetSpacing(), movingImage.GetOrigin(), movingImage.GetLargestPossibleRegion().GetSize()
    cMoving = [mOrigin[0] + mSpacing[0] * mSize[0] / 2.0,
               mOrigin[1] + mSpacing[1] * mSize[1] / 2.0]
    centerMoving = itk.Point[itk.D, 2](cMoving)

    transform.SetCenter(centerFixed)
    transform.SetTranslation(centerMoving - centerFixed)
    transform.SetAngle(0.0)
    registration.SetInitialTransformParameters(transform.GetParameters())

    optimizerScales = itk.OptimizerParameters[itk.D](transform.GetNumberOfParameters())
    translationScale = 1.0 / 1000.0
    optimizerScales[0] = 1.0
    optimizerScales[1] = translationScale
    optimizerScales[2] = translationScale
    optimizerScales[3] = translationScale
    optimizerScales[4] = translationScale
    optimizer.SetScales(optimizerScales)

    optimizer.SetRelaxationFactor(0.6)
    optimizer.SetMaximumStepLength(0.1)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(300)
    attach_iteration_logger(optimizer, "p3")

    registration.Update()
    print("  Optimizer stop:", registration.GetOptimizer().GetStopConditionDescription())

    finalParameters = registration.GetLastTransformParameters()
    finalAngle = finalParameters[0]
    print(f"  Angle (deg)  = {finalAngle * 180.0 / math.pi}")
    print(f"  Center       = ({finalParameters[1]}, {finalParameters[2]})")
    print(f"  Translation  = ({finalParameters[3]}, {finalParameters[4]})")
    print(f"  Iterations   = {optimizer.GetCurrentIteration()}")
    print(f"  Metric value = {optimizer.GetValue()}")

    finalTransform = TransformType.New()
    finalTransform.SetParameters(finalParameters)
    finalTransform.SetFixedParameters(transform.GetFixedParameters())

    resampler = itk.ResampleImageFilter[MovingImageType, FixedImageType].New()
    resampler.SetTransform(finalTransform)
    resampler.SetInput(movingReader.GetOutput())
    resampler.SetSize(fixedImage.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixedImage.GetOrigin())
    resampler.SetOutputSpacing(fixedImage.GetSpacing())
    resampler.SetOutputDirection(fixedImage.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    arr_fixed = itk.array_from_image(fixedReader.GetOutput()).astype(np.float32)
    arr_moving = itk.array_from_image(movingReader.GetOutput()).astype(np.float32)
    arr_registered = itk.array_from_image(resampler.GetOutput()).astype(np.float32)
    arr_diff_before = arr_fixed - arr_moving
    arr_diff = arr_fixed - arr_registered

    guardar_individuales(out_dir, arr_fixed, arr_moving, arr_registered, arr_diff)
    params_text = (
        "Transformación: CenteredRigid2DTransform   |   Métrica: MeanSquares   |   "
        "Optimizer: RegularStepGradientDescent\n"
        "Config: maxStep=0.10, minStep=0.001, maxIters=300, relaxation=0.6, "
        "translationScale=1/1000\n"
        f"Resultado: ángulo={finalAngle * 180.0 / math.pi:.4f}°   "
        f"centro=({finalParameters[1]:.2f}, {finalParameters[2]:.2f})   "
        f"T=({finalParameters[3]:.4f}, {finalParameters[4]:.4f})\n"
        f"           iters={optimizer.GetCurrentIteration()}   métrica={optimizer.GetValue():.4f}"
    )
    save_mosaic(
        [arr_fixed, arr_moving, arr_registered, arr_diff_before, arr_diff],
        "Pipeline 3: Rígido 2D",
        out_dir,
        params_text=params_text,
    )


def _extraer_slice_axial(itk_image_3d, z=90):
    Dimension = 3
    PixelType = itk.F
    InImageType = itk.Image[PixelType, Dimension]
    OutSliceType = itk.Image[PixelType, 2]

    extractor = itk.ExtractImageFilter[InImageType, OutSliceType].New()
    extractor.SetDirectionCollapseToSubmatrix()
    extractor.SetInput(itk_image_3d)

    region = itk_image_3d.GetLargestPossibleRegion()
    size = region.GetSize()
    start = region.GetIndex()
    size[2] = 0
    start[2] = z
    desired = region
    desired.SetSize(size)
    desired.SetIndex(start)
    extractor.SetExtractionRegion(desired)
    extractor.Update()
    return itk.array_from_image(extractor.GetOutput())


def pipeline4_rigido3D():
    print("\n=== Pipeline 4: Rígido 3D (VersorRigid3DTransform) ===")
    out_dir = str(OUT_ROOT / "pipeline4_rigido3D")
    os.makedirs(out_dir, exist_ok=True)

    fixed_path = str(IMG_DIR / "brainweb1e1a10f20.mha")
    moving_path = str(IMG_DIR / "brainweb1e1a10f20Rot10Tx15.mha")

    Dimension = 3
    PixelType = itk.F
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    TransformType = itk.VersorRigid3DTransform[itk.D]
    OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
    MetricType = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType]
    RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType]

    metric = MetricType.New()
    optimizer = OptimizerType.New()
    registration = RegistrationType.New()
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)

    initialTransform = TransformType.New()

    fixedReader = itk.ImageFileReader[FixedImageType].New()
    movingReader = itk.ImageFileReader[MovingImageType].New()
    fixedReader.SetFileName(fixed_path)
    movingReader.SetFileName(moving_path)
    fixedReader.Update()
    movingReader.Update()

    registration.SetFixedImage(fixedReader.GetOutput())
    registration.SetMovingImage(movingReader.GetOutput())

    initializer = itk.CenteredTransformInitializer[TransformType, FixedImageType, MovingImageType].New()
    initializer.SetTransform(initialTransform)
    initializer.SetFixedImage(fixedReader.GetOutput())
    initializer.SetMovingImage(movingReader.GetOutput())
    initializer.MomentsOn()
    initializer.InitializeTransform()

    rotation = itk.Versor[itk.D]()
    axis = itk.Vector[itk.D, 3]()
    axis.SetElement(0, 0.0)
    axis.SetElement(1, 0.0)
    axis.SetElement(2, 1.0)
    rotation.Set(axis, 0.0)
    initialTransform.SetRotation(rotation)
    registration.SetInitialTransform(initialTransform)

    optimizerScales = itk.OptimizerParameters[itk.D](initialTransform.GetNumberOfParameters())
    translationScale = 1.0 / 1000.0
    optimizerScales[0] = 1.0
    optimizerScales[1] = 1.0
    optimizerScales[2] = 1.0
    optimizerScales[3] = translationScale
    optimizerScales[4] = translationScale
    optimizerScales[5] = translationScale
    optimizer.SetScales(optimizerScales)
    optimizer.SetNumberOfIterations(200)
    optimizer.SetLearningRate(0.2)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetReturnBestParametersAndValue(True)
    attach_iteration_logger(optimizer, "p4")

    registration.SetNumberOfLevels(1)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])

    registration.Update()
    print("  Optimizer stop:", registration.GetOptimizer().GetStopConditionDescription())

    finalParameters = registration.GetOutput().Get().GetParameters()
    print(f"  Versor      = ({finalParameters[0]}, {finalParameters[1]}, {finalParameters[2]})")
    print(f"  Translation = ({finalParameters[3]}, {finalParameters[4]}, {finalParameters[5]})")
    print(f"  Iterations  = {optimizer.GetCurrentIteration()}")
    print(f"  Metric      = {optimizer.GetValue()}")

    finalTransform = TransformType.New()
    finalTransform.SetFixedParameters(registration.GetOutput().Get().GetFixedParameters())
    finalTransform.SetParameters(finalParameters)

    fixedImage = fixedReader.GetOutput()
    resampler = itk.ResampleImageFilter[MovingImageType, FixedImageType].New()
    resampler.SetTransform(finalTransform)
    resampler.SetInput(movingReader.GetOutput())
    resampler.SetSize(fixedImage.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixedImage.GetOrigin())
    resampler.SetOutputSpacing(fixedImage.GetSpacing())
    resampler.SetOutputDirection(fixedImage.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    arr_fixed = _extraer_slice_axial(fixedReader.GetOutput(), z=90)
    arr_moving = _extraer_slice_axial(movingReader.GetOutput(), z=90)
    arr_registered = _extraer_slice_axial(resampler.GetOutput(), z=90)
    arr_diff_before = arr_fixed - arr_moving
    arr_diff = arr_fixed - arr_registered

    guardar_individuales(out_dir, arr_fixed, arr_moving, arr_registered, arr_diff)
    params_text = (
        "Transformación: VersorRigid3DTransform   |   Métrica: MeanSquaresv4   |   "
        "Optimizer: RegularStepGradientDescentv4\n"
        "Inicialización: CenteredTransformInitializer (Moments)   "
        "translationScale=1/1000   niveles=1\n"
        "Config: learningRate=0.2, minStep=0.001, maxIters=200, "
        "ReturnBestParamsAndValue=True\n"
        f"Resultado: versor=({finalParameters[0]:.4e}, {finalParameters[1]:.4e}, "
        f"{finalParameters[2]:.4f})\n"
        f"           T=({finalParameters[3]:.4f}, {finalParameters[4]:.4f}, "
        f"{finalParameters[5]:.4f})   "
        f"iters={optimizer.GetCurrentIteration()}   métrica={optimizer.GetValue():.4f}"
    )
    save_mosaic(
        [arr_fixed, arr_moving, arr_registered, arr_diff_before, arr_diff],
        "Pipeline 4: Rígido 3D (slice axial z=90)",
        out_dir,
        params_text=params_text,
    )


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    pipeline1_traslacion2D_monomodal()
    pipeline2_traslacion2D_multimodal()
    pipeline3_rigido2D()
    pipeline4_rigido3D()

    print("\n✓ Archivos generados en output/:")
    output_str = str(OUT_ROOT)
    for root, dirs, files in os.walk(output_str):
        level = root.replace(output_str, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            print(f"{indent}  {f}")


if __name__ == "__main__":
    main()
