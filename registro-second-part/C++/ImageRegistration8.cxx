/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/


#include "itkImageRegistrationMethodv4.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkVersorRigid3DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkExtractImageFilter.h"

//  The following section of code implements a Command observer
//  that will monitor the evolution of the registration process.
//
#include "itkCommand.h"
class CommandIterationUpdate : public itk::Command
{
public:
  using Self = CommandIterationUpdate;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

protected:
  CommandIterationUpdate() = default;

public:
  using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
  using OptimizerPointer = const OptimizerType *;
  
  void Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }
  
  void Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    OptimizerPointer optimizer = static_cast<OptimizerPointer>(object);
    if (!itk::IterationEvent().CheckEvent(&event))
    {
      return;
    }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;
  }
};

int main(int argc, char * argv[])
{
  if (argc < 4)
  {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile  movingImageFile ";
    std::cerr << " outputImagefile  [differenceBeforeRegistration] ";
    std::cerr << " [differenceAfterRegistration] ";
    std::cerr << " [sliceBeforeRegistration] ";
    std::cerr << " [sliceDifferenceBeforeRegistration] ";
    std::cerr << " [sliceDifferenceAfterRegistration] ";
    std::cerr << " [sliceAfterRegistration] " << std::endl;
    return EXIT_FAILURE;
  }
  const unsigned int Dimension = 3;
  typedef float PixelType;
  typedef itk::Image<PixelType, Dimension> FixedImageType;
  typedef itk::Image<PixelType, Dimension> MovingImageType;

  typedef itk::VersorRigid3DTransform<double> TransformType;
  
  typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
  typedef itk::MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType> MetricType;
  typedef itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType, TransformType> RegistrationType;

  MetricType::Pointer metric = MetricType::New();
  OptimizerType::Pointer optimizer = OptimizerType::New();
  RegistrationType::Pointer registration = RegistrationType::New();

  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);

  TransformType::Pointer initialTransform = TransformType::New();
  
  typedef itk::ImageFileReader<FixedImageType> FixedImageReaderType;
  typedef itk::ImageFileReader<MovingImageType> MovingImageReaderType;
  FixedImageReaderType::Pointer fixedImageReader = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(argv[1]);
  movingImageReader->SetFileName(argv[2]);

  registration->SetFixedImage(fixedImageReader->GetOutput());
  registration->SetMovingImage(movingImageReader->GetOutput());

  typedef itk::CenteredTransformInitializer<TransformType,
                                      FixedImageType,
                                      MovingImageType> TransformInitializerType;
  TransformInitializerType::Pointer initializer = TransformInitializerType::New();
  
  initializer->SetTransform(initialTransform);
  initializer->SetFixedImage(fixedImageReader->GetOutput());
  initializer->SetMovingImage(movingImageReader->GetOutput());
  
  initializer->MomentsOn();
  initializer->InitializeTransform();
  
  typedef TransformType::VersorType VersorType;
  typedef VersorType::VectorType VectorType;
  VersorType rotation;
  VectorType axis;
  axis[0] = 0.0;
  axis[1] = 0.0;
  axis[2] = 1.0;
  const double angle = 0;
  rotation.Set(axis, angle);
  
  initialTransform->SetRotation(rotation);
  registration->SetInitialTransform(initialTransform);
  
  typedef OptimizerType::ScalesType OptimizerScalesType;
  OptimizerScalesType optimizerScales(
    initialTransform->GetNumberOfParameters());
  const double translationScale = 1.0 / 1000.0;
  optimizerScales[0] = 1.0;
  optimizerScales[1] = 1.0;
  optimizerScales[2] = 1.0;
  optimizerScales[3] = translationScale;
  optimizerScales[4] = translationScale;
  optimizerScales[5] = translationScale;
  optimizer->SetScales(optimizerScales);
  optimizer->SetNumberOfIterations(200);
  optimizer->SetLearningRate(0.2);
  optimizer->SetMinimumStepLength(0.001);
  optimizer->SetReturnBestParametersAndValue(true);

  // Create the Command observer and register it with the optimizer.
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver(itk::IterationEvent(), observer);

  // One level registration process without shrinking and smoothing.
  const unsigned int numberOfLevels = 1;

  RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize(1);
  shrinkFactorsPerLevel[0] = 1;

  RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize(1);
  smoothingSigmasPerLevel[0] = 0;

  registration->SetNumberOfLevels(numberOfLevels);
  registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
  registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

  try
  {
    registration->Update();
    std::cout << "Optimizer stop condition: "
              << registration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
  }
  catch (const itk::ExceptionObject & err)
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  const TransformType::ParametersType finalParameters =
    registration->GetOutput()->Get()->GetParameters();

  const double       versorX = finalParameters[0];
  const double       versorY = finalParameters[1];
  const double       versorZ = finalParameters[2];
  const double       finalTranslationX = finalParameters[3];
  const double       finalTranslationY = finalParameters[4];
  const double       finalTranslationZ = finalParameters[5];
  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
  const double       bestValue = optimizer->GetValue();

  // Print out results
  //
  std::cout << std::endl << std::endl;
  std::cout << "Result = " << std::endl;
  std::cout << " versor X      = " << versorX << std::endl;
  std::cout << " versor Y      = " << versorY << std::endl;
  std::cout << " versor Z      = " << versorZ << std::endl;
  std::cout << " Translation X = " << finalTranslationX << std::endl;
  std::cout << " Translation Y = " << finalTranslationY << std::endl;
  std::cout << " Translation Z = " << finalTranslationZ << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue << std::endl;

  TransformType::Pointer finalTransform = TransformType::New();

  finalTransform->SetFixedParameters(
    registration->GetOutput()->Get()->GetFixedParameters());
  finalTransform->SetParameters(finalParameters);

  TransformType::MatrixType matrix = finalTransform->GetMatrix();
  TransformType::OffsetType offset = finalTransform->GetOffset();
  std::cout << "Matrix = " << std::endl << matrix << std::endl;
  std::cout << "Offset = " << std::endl << offset << std::endl;

  typedef itk::ResampleImageFilter<MovingImageType, FixedImageType> ResampleFilterType;

  ResampleFilterType::Pointer resampler = ResampleFilterType::New();

  resampler->SetTransform(finalTransform);
  resampler->SetInput(movingImageReader->GetOutput());

  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();

  resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
  resampler->SetOutputOrigin(fixedImage->GetOrigin());
  resampler->SetOutputSpacing(fixedImage->GetSpacing());
  resampler->SetOutputDirection(fixedImage->GetDirection());
  resampler->SetDefaultPixelValue(100);

  typedef unsigned char OutputPixelType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::CastImageFilter<FixedImageType, OutputImageType> CastFilterType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;

  WriterType::Pointer writer = WriterType::New();
  CastFilterType::Pointer caster = CastFilterType::New();

  writer->SetFileName(argv[3]);

  caster->SetInput(resampler->GetOutput());
  writer->SetInput(caster->GetOutput());
  writer->Update();

  typedef itk::SubtractImageFilter<FixedImageType, FixedImageType, FixedImageType> DifferenceFilterType;
  DifferenceFilterType::Pointer difference = DifferenceFilterType::New();

  typedef itk::RescaleIntensityImageFilter<FixedImageType, OutputImageType> RescalerType;
  RescalerType::Pointer intensityRescaler = RescalerType::New();

  intensityRescaler->SetInput(difference->GetOutput());
  intensityRescaler->SetOutputMinimum(0);
  intensityRescaler->SetOutputMaximum(255);

  difference->SetInput1(fixedImageReader->GetOutput());
  difference->SetInput2(resampler->GetOutput());

  resampler->SetDefaultPixelValue(1);

  WriterType::Pointer writer2 = WriterType::New();
  writer2->SetInput(intensityRescaler->GetOutput());

  // Compute the difference image between the
  // fixed and resampled moving image.
  if (argc > 5)
  {
    writer2->SetFileName(argv[5]);
    writer2->Update();
  }

  typedef itk::IdentityTransform<double, Dimension> IdentityTransformType;
  IdentityTransformType::Pointer identity = IdentityTransformType::New();
  // Compute the difference image between the
  // fixed and moving image before registration.
  if (argc > 4)
  {
    resampler->SetTransform(identity);
    writer2->SetFileName(argv[4]);
    writer2->Update();
  }
  
  //  Here we extract slices from the input volume, and the difference volumes
  //  produced before and after the registration.
  typedef itk::Image<OutputPixelType, 2> OutputSliceType;
  typedef itk::ExtractImageFilter<OutputImageType, OutputSliceType> ExtractFilterType;
  ExtractFilterType::Pointer extractor = ExtractFilterType::New();
  extractor->SetDirectionCollapseToSubmatrix();
  extractor->InPlaceOn();

  FixedImageType::RegionType inputRegion =
    fixedImage->GetLargestPossibleRegion();
  FixedImageType::SizeType  size = inputRegion.GetSize();
  FixedImageType::IndexType start = inputRegion.GetIndex();

  // Select one slice as output
  size[2] = 0;
  start[2] = 90;
  FixedImageType::RegionType desiredRegion;
  desiredRegion.SetSize(size);
  desiredRegion.SetIndex(start);
  extractor->SetExtractionRegion(desiredRegion);
  typedef itk::ImageFileWriter<OutputSliceType> SliceWriterType;
  SliceWriterType::Pointer sliceWriter = SliceWriterType::New();
  sliceWriter->SetInput(extractor->GetOutput());
  if (argc > 6)
  {
    extractor->SetInput(caster->GetOutput());
    resampler->SetTransform(identity);
    sliceWriter->SetFileName(argv[6]);
    sliceWriter->Update();
  }
  if (argc > 7)
  {
    extractor->SetInput(intensityRescaler->GetOutput());
    resampler->SetTransform(identity);
    sliceWriter->SetFileName(argv[7]);
    sliceWriter->Update();
  }
  if (argc > 8)
  {
    resampler->SetTransform(finalTransform);
    sliceWriter->SetFileName(argv[8]);
    sliceWriter->Update();
  }
  if (argc > 9)
  {
    extractor->SetInput(caster->GetOutput());
    resampler->SetTransform(finalTransform);
    sliceWriter->SetFileName(argv[9]);
    sliceWriter->Update();
  }
  return EXIT_SUCCESS;
}