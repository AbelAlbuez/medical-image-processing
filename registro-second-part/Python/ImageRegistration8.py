#=========================================================================
#
#  Copyright NumFOCUS
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================*/


import itk
import sys

# Imágenes de entrada:
# Imagen fija: brainweb1e1a10f20.mha
# Imagen flotante: brainweb1e1a10f20Rot10Tx15.mha

if len(sys.argv) < 4 :
  print("Missing Parameters ")
  print("Usage: ", sys.argv[0], " fixedImageFile  movingImageFile outputImagefile  [differenceBeforeRegistration] [differenceAfterRegistration] [sliceBeforeRegistration] [sliceDifferenceBeforeRegistration] [sliceDifferenceAfterRegistration] [sliceAfterRegistration] ")
  sys.exit()

Dimension = 3
PixelType = itk.F
FixedImageType = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]

TransformType = itk.VersorRigid3DTransform[itk.D]
  
OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
MetricType = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType]
#RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType, TransformType]
RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType]

metric = MetricType.New()
optimizer = OptimizerType.New()
registration = RegistrationType.New()

registration.SetMetric(metric)
registration.SetOptimizer(optimizer)

initialTransform = TransformType.New()
  
FixedImageReaderType = itk.ImageFileReader[FixedImageType]
MovingImageReaderType = itk.ImageFileReader[MovingImageType]
fixedImageReader = FixedImageReaderType.New()
movingImageReader = MovingImageReaderType.New()

fixedImageReader.SetFileName(sys.argv[1])
movingImageReader.SetFileName(sys.argv[2])

registration.SetFixedImage(fixedImageReader.GetOutput())
registration.SetMovingImage(movingImageReader.GetOutput())

TransformInitializerType = itk.CenteredTransformInitializer[TransformType, FixedImageType, MovingImageType]
initializer = TransformInitializerType.New()
  
initializer.SetTransform(initialTransform)
initializer.SetFixedImage(fixedImageReader.GetOutput())
initializer.SetMovingImage(movingImageReader.GetOutput())
  
initializer.MomentsOn()
initializer.InitializeTransform()
  
rotation = itk.Versor[itk.D]()
axis = itk.Vector[itk.D, 3]()
axis.SetElement(0, 0.0)
axis.SetElement(1, 0.0)
axis.SetElement(2, 1.0)
angle = 0.0
rotation.Set(axis, angle)

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

# Proceso de registro de un nivel sin encoger ni suavizar
numberOfLevels = 1;

shrinkFactorsPerLevel = list()
shrinkFactorsPerLevel.append(1)

smoothingSigmasPerLevel = list()
smoothingSigmasPerLevel.append(0)

registration.SetNumberOfLevels(numberOfLevels)
registration.SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel)
registration.SetShrinkFactorsPerLevel(shrinkFactorsPerLevel)

registration.Update()
print("Optimizer stop condition: ", registration.GetOptimizer().GetStopConditionDescription())
  
finalParameters = registration.GetOutput().Get().GetParameters()

versorX = finalParameters[0]
versorY = finalParameters[1]
versorZ = finalParameters[2]
finalTranslationX = finalParameters[3]
finalTranslationY = finalParameters[4]
finalTranslationZ = finalParameters[5]
numberOfIterations = optimizer.GetCurrentIteration()
bestValue = optimizer.GetValue()

# Imprimir resultados
print("Result = ")
print(" versor X      = ", versorX)
print(" versor Y      = ", versorY)
print(" versor Z      = ", versorZ)
print(" Translation X = ", finalTranslationX)
print(" Translation Y = ", finalTranslationY)
print(" Translation Z = ", finalTranslationZ)
print(" Iterations    = ", numberOfIterations)
print(" Metric value  = ", bestValue)

finalTransform = TransformType.New()

finalTransform.SetFixedParameters(registration.GetOutput().Get().GetFixedParameters())
finalTransform.SetParameters(finalParameters)

matrix = finalTransform.GetMatrix()
offset = finalTransform.GetOffset()
print("Matrix = ")
print(matrix)
print("Offset = ")
print(offset)

ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]

resampler = ResampleFilterType.New()
resampler.SetTransform(finalTransform)
resampler.SetInput(movingImageReader.GetOutput())

fixedImage = fixedImageReader.GetOutput()

resampler.SetSize(fixedImage.GetLargestPossibleRegion().GetSize())
resampler.SetOutputOrigin(fixedImage.GetOrigin())
resampler.SetOutputSpacing(fixedImage.GetSpacing())
resampler.SetOutputDirection(fixedImage.GetDirection())
resampler.SetDefaultPixelValue(100)

OutputPixelType = itk.UC
OutputImageType = itk.Image[OutputPixelType, Dimension]
CastFilterType = itk.CastImageFilter[FixedImageType, OutputImageType]
WriterType = itk.ImageFileWriter[OutputImageType]

writer = WriterType.New()
caster = CastFilterType.New()

writer.SetFileName(sys.argv[3])

caster.SetInput(resampler.GetOutput())
writer.SetInput(caster.GetOutput())
writer.Update()

DifferenceFilterType = itk.SubtractImageFilter[FixedImageType, FixedImageType, FixedImageType]
difference = DifferenceFilterType.New()

RescalerType = itk.RescaleIntensityImageFilter[FixedImageType, OutputImageType]
intensityRescaler = RescalerType.New()

intensityRescaler.SetInput(difference.GetOutput())
intensityRescaler.SetOutputMinimum(0)
intensityRescaler.SetOutputMaximum(255)

difference.SetInput1(fixedImageReader.GetOutput())
difference.SetInput2(resampler.GetOutput())

resampler.SetDefaultPixelValue(1)

writer2 = WriterType.New()
writer2.SetInput(intensityRescaler.GetOutput())

# Calcular la imagen de diferencias entre la imagen fija y la movible remuestreada
if len(sys.argv) > 5 :
  writer2.SetFileName(sys.argv[5])
  writer2.Update()

IdentityTransformType = itk.IdentityTransform[itk.D, Dimension]
identity = IdentityTransformType.New()
# Calcular la imagen de diferencias entre la imagen fija y la movible antes del registro
if len(sys.argv) > 4 :
  resampler.SetTransform(identity)
  writer2.SetFileName(sys.argv[4])
  writer2.Update()

# Aquí se extraen cortes del volumen de entrada y de los volúmenes de diferencias
# producidos antes y después del proceso de registro
OutputSliceType = itk.Image[OutputPixelType, 2]
ExtractFilterType = itk.ExtractImageFilter[OutputImageType, OutputSliceType]
extractor = ExtractFilterType.New()
extractor.SetDirectionCollapseToSubmatrix()
extractor.InPlaceOn()

inputRegion = fixedImage.GetLargestPossibleRegion()
size = inputRegion.GetSize()
start = inputRegion.GetIndex()

# Seleccionar un corte como salida
size[2] = 0
start[2] = 90
#desiredRegion = FixedImageType.RegionType.New()
desiredRegion = inputRegion
desiredRegion.SetSize(size)
desiredRegion.SetIndex(start)
extractor.SetExtractionRegion(desiredRegion)
SliceWriterType = itk.ImageFileWriter[OutputSliceType]
sliceWriter = SliceWriterType.New()
sliceWriter.SetInput(extractor.GetOutput())
if len(sys.argv) > 6 :
  extractor.SetInput(caster.GetOutput())
  resampler.SetTransform(identity)
  sliceWriter.SetFileName(sys.argv[6])
  sliceWriter.Update()

if len(sys.argv) > 7 :
  extractor.SetInput(intensityRescaler.GetOutput())
  resampler.SetTransform(identity)
  sliceWriter.SetFileName(sys.argv[7])
  sliceWriter.Update()

if len(sys.argv) > 8 :
  resampler.SetTransform(finalTransform)
  sliceWriter.SetFileName(sys.argv[8])
  sliceWriter.Update()

if len(sys.argv) > 9 :
  extractor.SetInput(caster.GetOutput())
  resampler.SetTransform(finalTransform)
  sliceWriter.SetFileName(sys.argv[9])
  sliceWriter.Update()
