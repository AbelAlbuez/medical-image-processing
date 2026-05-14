#=========================================================================
#
#  Program:   Insight Segmentation & Registration Toolkit
#  Module:    $RCSfile: ImageRegistration5.cxx,v $
#  Language:  C++
#  Date:      $Date: 2009/06/24 12:09:13 $
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
import math

# Imágenes de entrada:
# Imagen fija: BrainProtonDensitySliceBorder20.png
# Imagen flotante: BrainProtonDensitySliceR10X13Y17.png

# identificar la cantidad de parámetros de entrada del comando
# si no son suficientes, mostrar el uso del programa
if len(sys.argv) < 4 :
  print("Missing Parameters ")
  print("Usage: ", argv[0], " fixedImageFile  movingImageFile outputImagefile  [differenceAfterRegistration] [differenceBeforeRegistration] [initialStepLength] ")
  sys.exit()
  
# Definición de la dimensión y tipo de pixel de la imagen
# así como de las imágenes de entrada (fija y flotante)
Dimension = 2
PixelType = itk.UC

FixedImageType = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]

# Definición del tipo de transformación, sólo requiere el tipo de dato
# de las coordenadas del espacio
TransformType = itk.CenteredRigid2DTransform[itk.D]

# Definición de los componentes del método de registro
OptimizerType = itk.RegularStepGradientDescentOptimizer
MetricType = itk.MeanSquaresImageToImageMetric[FixedImageType, MovingImageType]
InterpolatorType = itk.LinearInterpolateImageFunction[MovingImageType, itk.D]
RegistrationType = itk.ImageRegistrationMethod[FixedImageType, MovingImageType]

metric        = MetricType.New()
optimizer     = OptimizerType.New()
interpolator  = InterpolatorType.New()
registration  = RegistrationType.New()
  
registration.SetMetric(        metric        )
registration.SetOptimizer(     optimizer     )
registration.SetInterpolator(  interpolator  )

# Se construye la transformación y se conecta al método de registro
transform = TransformType.New()
registration.SetTransform( transform )

# Lectura de las imágenes de entrada y conexión al método de registro
FixedImageReaderType = itk.ImageFileReader[FixedImageType]
MovingImageReaderType = itk.ImageFileReader[MovingImageType]

fixedImageReader  = FixedImageReaderType.New()
movingImageReader = MovingImageReaderType.New()

fixedImageReader.SetFileName(  sys.argv[1] )
movingImageReader.SetFileName( sys.argv[2] )

registration.SetFixedImage(    fixedImageReader.GetOutput()    )
registration.SetMovingImage(   movingImageReader.GetOutput()   )
fixedImageReader.Update()

registration.SetFixedImageRegion( fixedImageReader.GetOutput().GetBufferedRegion() )

# Se actualizan los lectores para garantizar que los parámetros de las
# imágenes son válidos al momento de inicializar la transformación
fixedImageReader.Update()
movingImageReader.Update()

# Como es una transformación centrada, se calcula el centro de la imagen fija
fixedImage = fixedImageReader.GetOutput()

fixedSpacing = fixedImage.GetSpacing()
fixedOrigin  = fixedImage.GetOrigin()
fixedRegion  = fixedImage.GetLargestPossibleRegion()
fixedSize    = fixedRegion.GetSize()
  
cFixed = list()
cFixed.append(fixedOrigin[0] + fixedSpacing[0] * fixedSize[0] / 2.0)
cFixed.append(fixedOrigin[1] + fixedSpacing[1] * fixedSize[1] / 2.0)
centerFixed = itk.Point[itk.D,2](cFixed)

# Y de manera similar se calcula el centro de la imagen flotante
movingImage = movingImageReader.GetOutput()

movingSpacing = movingImage.GetSpacing()
movingOrigin  = movingImage.GetOrigin()
movingRegion  = movingImage.GetLargestPossibleRegion()
movingSize    = movingRegion.GetSize()
  
cMoving = list()
cMoving.append(movingOrigin[0] + movingSpacing[0] * movingSize[0] / 2.0)
cMoving.append(movingOrigin[1] + movingSpacing[1] * movingSize[1] / 2.0)
centerMoving = itk.Point[itk.D,2](cMoving)

# La transformación se inicializa en este caso fijando su centro como el de la 
# imagen fija, y fijando la traslación como la diferencia entre los centros de 
# las imágenes
transform.SetCenter( centerFixed )
transform.SetTranslation( centerMoving - centerFixed )

# La rotación se inicializa con un ángulo de cero grados
transform.SetAngle( 0.0 );

# Los parámetros actuales, que acabamos de fijar, se utilizan para
# inicializar la transformación como se hacía en los anteriores ejemplos
registration.SetInitialTransformParameters( transform.GetParameters() )

# El optimizador provee una funcionalidad de reescalamiento, teniendo en cuenta
# que las unidades de rotación y traslación son diferentes
optimizerScales = itk.OptimizerParameters[itk.D](transform.GetNumberOfParameters())
translationScale = 1.0 / 1000.0

optimizerScales[0] = 1.0
optimizerScales[1] = translationScale
optimizerScales[2] = translationScale
optimizerScales[3] = translationScale
optimizerScales[4] = translationScale

optimizer.SetScales( optimizerScales )

# A continuación hay que fijar los parámetros del método de optimización
initialStepLength = 0.1

if len(sys.argv) > 6 :
  initialStepLength = float( sys.argv[6] )

optimizer.SetRelaxationFactor( 0.6 )
optimizer.SetMaximumStepLength( initialStepLength )
optimizer.SetMinimumStepLength( 0.001 )
optimizer.SetNumberOfIterations( 300 )

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

finalAngle           = finalParameters[0]
finalRotationCenterX = finalParameters[1]
finalRotationCenterY = finalParameters[2]
finalTranslationX    = finalParameters[3]
finalTranslationY    = finalParameters[4]

numberOfIterations = optimizer.GetCurrentIteration()

bestValue = optimizer.GetValue()
 
# Imprimir los resultados
finalAngleInDegrees = finalAngle * 180.0 / math.pi

print("Result = ")
print(" Angle (radians)   = ", finalAngle)
print(" Angle (degrees)   = ", finalAngleInDegrees)
print(" Center X      = ", finalRotationCenterX)
print(" Center Y      = ", finalRotationCenterY)
print(" Translation X = ", finalTranslationX)
print(" Translation Y = ", finalTranslationY)
print(" Iterations    = ", numberOfIterations)
print(" Metric value  = ", bestValue)

# Aplicar la transformación obtenida a la imagen flotante
ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]

finalTransform = TransformType.New()
finalTransform.SetParameters( finalParameters )
finalTransform.SetFixedParameters( transform.GetFixedParameters() )

resample = ResampleFilterType.New()
resample.SetTransform( finalTransform )
resample.SetInput( movingImageReader.GetOutput() )
resample.SetSize( fixedImage.GetLargestPossibleRegion().GetSize() )
resample.SetOutputOrigin(  fixedImage.GetOrigin() )
resample.SetOutputSpacing( fixedImage.GetSpacing() )
resample.SetOutputDirection( fixedImage.GetDirection() )
resample.SetDefaultPixelValue( 100 )
  
# Definir el filtro de escritura de la imagen transformada
WriterFixedType = itk.ImageFileWriter[FixedImageType]

writer =  WriterFixedType.New()
writer.SetFileName( sys.argv[3] )
writer.SetInput( resample.GetOutput()   )
writer.Update()

# Generar la imagen diferencia utilizando el filtro de sustracción
DifferenceImageType = itk.Image[itk.F, Dimension]

InCastFilterType = itk.CastImageFilter[FixedImageType, DifferenceImageType]
in1caster = InCastFilterType.New()
in1caster.SetInput( fixedImageReader.GetOutput() )

in2caster = InCastFilterType.New()
in2caster.SetInput( resample.GetOutput() )

DifferenceFilterType = itk.SubtractImageFilter[DifferenceImageType, DifferenceImageType, DifferenceImageType]

difference = DifferenceFilterType.New()

OutputPixelType = itk.UC

OutputImageType = itk.Image[OutputPixelType, Dimension]
  
RescalerType = itk.RescaleIntensityImageFilter[DifferenceImageType, OutputImageType]

intensityRescaler = RescalerType.New()
intensityRescaler.SetOutputMinimum(   0 )
intensityRescaler.SetOutputMaximum( 255 )

difference.SetInput1( in1caster.GetOutput() )
difference.SetInput2( in2caster.GetOutput() )

resample.SetDefaultPixelValue( 1 )

intensityRescaler.SetInput( difference.GetOutput() ) 
  
# Definir el filtro de escritura de la imagen diferencia
WriterType = itk.ImageFileWriter[OutputImageType]

writer2 =  WriterType.New()
writer2.SetInput( intensityRescaler.GetOutput() )

# Escribir la imagen diferencia después del registro
if len(sys.argv) > 4 :
  writer2.SetFileName( sys.argv[4] )
  writer2.Update()

# Calcular y escribir la imagen diferencia antes del registro
identityTransform = TransformType.New()
identityTransform.SetIdentity()
resample.SetTransform( identityTransform )
if len(sys.argv) > 5 :
  writer2.SetFileName( sys.argv[5] )
  writer2.Update()
