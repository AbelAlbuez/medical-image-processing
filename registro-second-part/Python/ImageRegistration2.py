#=========================================================================
#
#  Program:   Insight Segmentation & Registration Toolkit
#  Module:    $RCSfile: ImageRegistration2.cxx,v $
#  Language:  C++
#  Date:      $Date: 2009/06/24 12:09:08 $
#  Version:   $Revision: 1.47 $
#
#  Copyright (c) Insight Software Consortium. All rights reserved.
#  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.
#
#     This software is distributed WITHOUT ANY WARRANTY; without even 
#     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
#     PURPOSE.  See the above copyright notices for more information.
#
#=========================================================================*/
import itk
import sys

# Imágenes de entrada:
# Imagen fija: BrainT1SliceBorder20.png
# Imagen flotante: BrainProtonDensitySliceShifted13x17y.png

# identificar la cantidad de parámetros de entrada del comando
# si no son suficientes, mostrar el uso del programa
if len(sys.argv) < 4 :
  print("Missing Parameters ")
  print("Usage: ", argv[0], " fixedImageFile movingImageFile outputImagefile [checkerBoardBefore] [checkerBoardAfter]")
  sys.exit()
  
# Definición de la dimensión y tipo de pixel de la imagen
# así como de las imágenes de entrada (fija y flotante)
Dimension = 2;
PixelType = itk.US;
  
FixedImageType = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]

# Para trabajar con información mutua, se define un nuevo tipo interno
# en el que se normalizarán las imágenes de entrada
InternalPixelType = itk.F
InternalImageType = itk.Image[InternalPixelType, Dimension]

# Definición de los elementos de la transformación
TransformType = itk.TranslationTransform[itk.D, Dimension]
OptimizerType = itk.GradientDescentOptimizer         
InterpolatorType = itk.LinearInterpolateImageFunction[InternalImageType, itk.D]
RegistrationType = itk.ImageRegistrationMethod[InternalImageType, InternalImageType]

# Definición de la métrica usando información mutua
MetricType = itk.MutualInformationImageToImageMetric[InternalImageType, InternalImageType]

transform     = TransformType.New()
optimizer     = OptimizerType.New()
interpolator  = InterpolatorType.New()
registration  = RegistrationType.New()

registration.SetOptimizer(     optimizer     )
registration.SetTransform(     transform     )
registration.SetInterpolator(  interpolator  )

# La métrica se inicializa con el método New() y se conecta al método de 
# registro
metric        = MetricType.New()
registration.SetMetric( metric  )

# La métrica requiere la definición de varios parámetros
metric.SetFixedImageStandardDeviation(  0.4 )
metric.SetMovingImageStandardDeviation( 0.4 )

# Lectura de las imágenes de entrada
FixedImageReaderType = itk.ImageFileReader[FixedImageType]
MovingImageReaderType = itk.ImageFileReader[MovingImageType]

fixedImageReader  = FixedImageReaderType.New()
movingImageReader = MovingImageReaderType.New()

fixedImageReader.SetFileName(  sys.argv[1] )
movingImageReader.SetFileName( sys.argv[2] )

# Se utilizan filtros de normalización para las imágenes de entrada, con el
# tipo de dato interno definido anteriormente. Esto normaliza la distribución
# estadística de las dos imágenes
In1CastFilterType = itk.CastImageFilter[FixedImageType, InternalImageType]
in1caster = In1CastFilterType.New()
in1caster.SetInput( fixedImageReader.GetOutput() )

In2CastFilterType = itk.CastImageFilter[MovingImageType, InternalImageType]
in2caster = In2CastFilterType.New()
in2caster.SetInput( movingImageReader.GetOutput() )

FixedNormalizeFilterType = itk.NormalizeImageFilter[InternalImageType, InternalImageType]
MovingNormalizeFilterType = itk.NormalizeImageFilter[InternalImageType, InternalImageType]

fixedNormalizer = FixedNormalizeFilterType.New()
movingNormalizer = MovingNormalizeFilterType.New()

# Adicionalmente, las imágenes se pasan por un filtro pasabajos (suavizado)
# para incrementar la robustez al ruido
GaussianFilterType = itk.DiscreteGaussianImageFilter[InternalImageType, InternalImageType]
  
fixedSmoother  = GaussianFilterType.New()
movingSmoother = GaussianFilterType.New()

fixedSmoother.SetVariance( 2.0 )
movingSmoother.SetVariance( 2.0 )
  
# La salida de los lectores se conecta a los filtros de normalización
# La salida de la normalización se conecta a los filtros de suavizado
# La salida de los filtros de suavizado se conecta al método de registro
fixedNormalizer.SetInput(  in1caster.GetOutput() )
movingNormalizer.SetInput( in2caster.GetOutput() )

fixedSmoother.SetInput( fixedNormalizer.GetOutput() )
movingSmoother.SetInput( movingNormalizer.GetOutput() )

registration.SetFixedImage( fixedSmoother.GetOutput() )
registration.SetMovingImage( movingSmoother.GetOutput() )

fixedNormalizer.Update()
fixedImageRegion = fixedNormalizer.GetOutput().GetBufferedRegion()
registration.SetFixedImageRegion( fixedImageRegion )

# Se definen los parámetros iniciales de la transformación
initialParameters = itk.OptimizerParameters[itk.D](transform.GetNumberOfParameters())
initialParameters[0] = 0.0     # Initial offset in mm along X
initialParameters[1] = 0.0     # Initial offset in mm along Y
  
registration.SetInitialTransformParameters( initialParameters )

# Definción del número de muestras espaciales a tomar de la imagen para la métrica
numberOfPixels = fixedImageRegion.GetNumberOfPixels()
  
numberOfSamples = int(numberOfPixels * 0.01)

metric.SetNumberOfSpatialSamples( numberOfSamples )

# La función de costo del optimizador en este caso se maximiza, ya que mayores
# valores de información mutua indican mejores ajustes
optimizer.SetLearningRate( 15.0 )
optimizer.SetNumberOfIterations( 200 )
optimizer.MaximizeOn()

# Se fija el comando/observador, para seguir el progreso del registro
def iteration_update():
  metric_value = optimizer.GetValue()
  current_parameters = optimizer.GetCurrentPosition()
  parameter_list = [current_parameters[i] for i in range(len(current_parameters))]
  current_iteration = optimizer.GetCurrentIteration()
  print(current_iteration, " = ", metric_value, " : ", parameter_list)

iteration_command = itk.PyCommand.New()
iteration_command.SetCommandCallable(iteration_update)
optimizer.AddObserver(itk.IterationEvent(), iteration_command)

registration.Update();
print("Optimizer stop condition: ", registration.GetOptimizer().GetStopConditionDescription())

# Obtener los parámetros finales del registro
finalParameters = registration.GetLastTransformParameters()
  
TranslationAlongX = finalParameters[0]
TranslationAlongY = finalParameters[1]
  
numberOfIterations = optimizer.GetCurrentIteration()
  
bestValue = optimizer.GetValue()

# Imprimir los resultados
print("Result = ")
print(" Translation X = ", TranslationAlongX)
print(" Translation Y = ", TranslationAlongY)
print(" Iterations    = ", numberOfIterations)
print(" Metric value  = ", bestValue)
print(" Numb. Samples = ", numberOfSamples)

# Aplicar la transformación obtenida a la imagen flotante
ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]    

finalTransform = TransformType.New()
finalTransform.SetParameters( finalParameters )
finalTransform.SetFixedParameters( transform.GetFixedParameters() )

resample = ResampleFilterType.New()
resample.SetTransform( finalTransform )
resample.SetInput( movingImageReader.GetOutput() )
  
fixedImage = fixedImageReader.GetOutput()

resample.SetSize( fixedImage.GetLargestPossibleRegion().GetSize() )
resample.SetOutputOrigin(  fixedImage.GetOrigin() )
resample.SetOutputSpacing( fixedImage.GetSpacing() )
resample.SetOutputDirection( fixedImage.GetDirection() )
resample.SetDefaultPixelValue( 100 )

# Escribir la imagen resultado
OutputPixelType = itk.UC

OutputImageType = itk.Image[OutputPixelType, Dimension]
  
CastFilterType = itk.CastImageFilter[FixedImageType, OutputImageType]
                    
WriterType = itk.ImageFileWriter[OutputImageType]

writer =  WriterType.New()
caster =  CastFilterType.New()

writer.SetFileName( sys.argv[3] )
  
caster.SetInput( resample.GetOutput() )
writer.SetInput( caster.GetOutput()   )
writer.Update()

# Otra forma de comparar los resultados es poniendo de forma alternada pedazos 
# de una imagen y de otra, como los cuadros blancos y negros de un ajedrez
# Esto se logra con un filtro CheckerBoard (ajedrez en inglés)
CheckerBoardFilterType = itk.CheckerBoardImageFilter[FixedImageType]
checker = CheckerBoardFilterType.New()
checker.SetInput1( fixedImage )
checker.SetInput2( resample.GetOutput() )

caster.SetInput( checker.GetOutput() )
writer.SetInput( caster.GetOutput() )
  
# Generar resultados de comparación antes de aplicar la transformación
identityTransform = TransformType.New()
identityTransform.SetIdentity()
resample.SetTransform( identityTransform )

if len(sys.argv) > 4 :
  writer.SetFileName( sys.argv[4] )
  writer.Update()
 
# Generar resultados de comparación después de aplicar la transformación
resample.SetTransform( finalTransform )
if len(sys.argv) > 5 :
  writer.SetFileName( sys.argv[5] )
  writer.Update()
