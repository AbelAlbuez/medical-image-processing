#=========================================================================
#
#  Program:   Insight Segmentation & Registration Toolkit
#  Module:    $RCSfile: ImageRegistration1.cxx,v $
#  Language:  C++
#  Date:      $Date: 2007/11/22 00:30:16 $
#  Version:   $Revision: 1.53 $
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
# Imagen fija: BrainProtonDensitySliceBorder20.png
# Imagen flotante: BrainProtonDensitySliceShifted13x17y.png

# identificar la cantidad de parámetros de entrada del comando
# si no son suficientes, mostrar el uso del programa
if len(sys.argv) < 4 :
  print("Missing Parameters ")
  print("Usage: ", sys.argv[0], " fixedImageFile movingImageFile outputImagefile [differenceImageAfter] [differenceImageBefore]")
  sys.exit()

# Identificar la dimensión de la imagen y el tipo de dato a usar
Dimension = 2;
PixelType = itk.F;

# Definición de los tipos de las imágenes de entrada
FixedImageType = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]

# Definición de la transformación que lleva una imagen flotante a la imagen fija
TransformType = itk.TranslationTransform[itk.D, Dimension]

# Definición del optimizador que explora el espacio de parámetros de la
# transformación en busca de los valores óptimos de la métrica
# Optimizador de Descenso por Gradiente con Paso Regular
OptimizerType = itk.RegularStepGradientDescentOptimizer

# Definición de la métrica que identifica la similaridad entre las imágenes
# Error cuadrático medio
MetricType = itk.MeanSquaresImageToImageMetric[FixedImageType, MovingImageType]

# Definición del interpolador, que evalúa las intensidades de la imagen
# flotante en posiciones no ajustadas a la grilla
# Función de Interpolación Lineal de Imágenes
InterpolatorType = itk.LinearInterpolateImageFunction[MovingImageType, itk.D] 

# Definición del método de registro, que interconecta los elementos
# definidos hasta ahora
RegistrationType = itk.ImageRegistrationMethod[FixedImageType, MovingImageType]    ;

# Cada uno de los componentes del método de registro se crea utilizando
# su método New()
metric       = MetricType.New()
transform    = TransformType.New()
optimizer    = OptimizerType.New()
interpolator = InterpolatorType.New()
registration = RegistrationType.New()

# Cada componente se conecta con el método de registro
registration.SetMetric(      metric      )
registration.SetOptimizer(   optimizer   )
registration.SetTransform(   transform   )
registration.SetInterpolator(interpolator)

# Las imágenes de entrada (fija y flotante) se leen de archivos 
# externos utilizando clases lectoras (readers), utilizando los 
# parámetros del programa como los nombres de archivo
FixedImageReaderType = itk.ImageFileReader[FixedImageType]
MovingImageReaderType = itk.ImageFileReader[MovingImageType]
fixedImageReader  = FixedImageReaderType.New()
movingImageReader = MovingImageReaderType.New()

fixedImageReader.SetFileName(sys.argv[1])
movingImageReader.SetFileName(sys.argv[2])

# Las imágenes se asignan al método de registro
registration.SetFixedImage(fixedImageReader.GetOutput())
registration.SetMovingImage(movingImageReader.GetOutput())

# El registro puede realizarse sólo en una parte de la imagen, en este
# caso se usa el contenido disponible de la imagen como región de registro
fixedImageReader.Update()
registration.SetFixedImageRegion(fixedImageReader.GetOutput().GetBufferedRegion())

# Definición de los parámetros iniciales de la transformación
# en este caso se define una traslación en ceros (transformación identidad)
initialParameters = itk.OptimizerParameters[itk.D](transform.GetNumberOfParameters())
initialParameters[0] = 0.0     # Initial offset in mm along X
initialParameters[1] = 0.0     # Initial offset in mm along Y
  
registration.SetInitialTransformParameters(initialParameters)

# Definición de algunos parámetros del optimizador
optimizer.SetMaximumStepLength(4.00)  
optimizer.SetMinimumStepLength(0.01)
optimizer.SetNumberOfIterations(200)

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

# El proceso de registro se ejecuta haciendo un llamado al método Update()
# del filtro. 
registration.Update()

# El resultado del proceso de registro es un conjunto de parámetros que definen
# de manera única la transformación. El optimizador puede retornar también
# la cantidad de iteraciones realizadas y el valor de la métrica.
finalParameters = registration.GetLastTransformParameters()

TranslationAlongX = finalParameters[0]
TranslationAlongY = finalParameters[1]

numberOfIterations = optimizer.GetCurrentIteration()
bestValue = optimizer.GetValue()

print("Result = ")
print(" Translation X = ", TranslationAlongX)
print(" Translation Y = ", TranslationAlongY)
print(" Iterations    = ", numberOfIterations)
print(" Metric value  = ", bestValue)

# Hasta aquí, sólo se tienen los parámetros de la transformación.
# Usualmente, se aplica la transformación a la imagen flotante para verla transformada
# Esto se realiza a través de un filtro de remuestreo de la imagen
# Debido a que después de aplicar la transformación, la imagen queda en coordenadas continuas
# y hay que volver a muestrearla en una grilla discreta para poder visualizarla o guardarla 
# como imagen.
ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]
  
resampler = ResampleFilterType.New()
resampler.SetInput(movingImageReader.GetOutput())
resampler.SetTransform(registration.GetOutput().Get())

# la imagen fija define el sistema de referencia
fixedImage = fixedImageReader.GetOutput();
resampler.SetSize( fixedImage.GetLargestPossibleRegion().GetSize() )
resampler.SetOutputOrigin(  fixedImage.GetOrigin() )
resampler.SetOutputSpacing( fixedImage.GetSpacing() )
resampler.SetOutputDirection( fixedImage.GetDirection() )
resampler.SetDefaultPixelValue( 100 )
  
# La salida del filtro de remuestreo se pasa a un filtro de escritura que genera la
# imagen en disco, utilizando un conversor de tipo de dato
OutputPixelType = itk.UC
OutputImageType = itk.Image[OutputPixelType, Dimension]
CastFilterType = itk.CastImageFilter[FixedImageType, OutputImageType]
WriterType = itk.ImageFileWriter[OutputImageType]
  
writer =  WriterType.New()
caster =  CastFilterType.New()
  
writer.SetFileName( sys.argv[3] )

caster.SetInput( resampler.GetOutput() )
writer.SetInput( caster.GetOutput()   )
writer.Update()

# La imagen fija y la flotante (transformada) pueden compararse entre sí por medio
# de un filtro diferencia (sustracción)
# Input 1: Imagen A (fija)
# Input 2: Imagen B (transformada o de comparación)
# Output: Resultado de la resta (imagen diferencia)
DifferenceFilterType = itk.SubtractImageFilter[FixedImageType, FixedImageType, FixedImageType]
difference = DifferenceFilterType.New()

difference.SetInput1( fixedImageReader.GetOutput() )
difference.SetInput2( resampler.GetOutput() )

# Las diferencias pueden ser muy pequeñas para ser visualizadas, por lo que se
# pasan a través de un filtro que reescala los rangos de intensidades
RescalerType = itk.RescaleIntensityImageFilter[FixedImageType, OutputImageType]
intensityRescaler = RescalerType.New()
  
intensityRescaler.SetInput( difference.GetOutput() )
intensityRescaler.SetOutputMinimum(   0 )
intensityRescaler.SetOutputMaximum( 255 )

# valor que se asigna a los píxeles “vacíos” o fuera de la imagen cuando se aplica 
# el remuestreo
resampler.SetDefaultPixelValue( 1 )

# esta diferencia puede escribirse a un archivo
writer2 = WriterType.New()
writer2.SetInput( intensityRescaler.GetOutput() );  

if len(sys.argv) > 4 :
  writer2.SetFileName( sys.argv[4] )
  writer2.Update()

# Las diferencias antes de la transformación pueden establecerse aplicando el filtro
# de diferencias a la imagen flotante con una transformación identidad 
identityTransform = TransformType.New()
identityTransform.SetIdentity()
resampler.SetTransform( identityTransform )

if len(sys.argv) > 5 :
  writer2.SetFileName( sys.argv[5] )
  writer2.Update()
