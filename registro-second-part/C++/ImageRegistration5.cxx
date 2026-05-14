/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: ImageRegistration5.cxx,v $
  Language:  C++
  Date:      $Date: 2009/06/24 12:09:13 $
  Version:   $Revision: 1.47 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "itkImageRegistrationMethod.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkImage.h"

#include "itkCenteredRigid2DTransform.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

#include "itkCommand.h"

// Implementación del comando/observador que monitorea la evolución
// del proceso de registro
class CommandIterationUpdate : public itk::Command 
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro( Self );
protected:
  CommandIterationUpdate() {};
public:
  typedef itk::RegularStepGradientDescentOptimizer  OptimizerType;
  typedef   const OptimizerType *                   OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
    Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
    OptimizerPointer optimizer = 
      dynamic_cast< OptimizerPointer >( object );
    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
      return;
      }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;
    }
};

int main( int argc, char *argv[] )
{
  //Imágenes de entrada:
  // Imagen fija: BrainProtonDensitySliceBorder20.png
  // Imagen flotante: BrainProtonDensitySliceR10X13Y17.png

  // identificar la cantidad de parámetros de entrada del comando
  // si no son suficientes, mostrar el uso del programa
  if( argc < 4 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile  movingImageFile ";
    std::cerr << " outputImagefile  [differenceAfterRegistration] ";
    std::cerr << " [differenceBeforeRegistration] ";
    std::cerr << " [initialStepLength] "<< std::endl;
    return EXIT_FAILURE;
    }
  
  // Definición de la dimensión y tipo de pixel de la imagen
  // así como de las imágenes de entrada (fija y flotante)
  const    unsigned int    Dimension = 2;
  typedef  unsigned char   PixelType;

  typedef itk::Image< PixelType, Dimension >  FixedImageType;
  typedef itk::Image< PixelType, Dimension >  MovingImageType;

  // Definición del tipo de transformación, sólo requiere el tipo de dato
  // de las coordenadas del espacio
  typedef itk::CenteredRigid2DTransform< double > TransformType;

  // Definición de los componentes del método de registro
  typedef itk::RegularStepGradientDescentOptimizer       OptimizerType;
  typedef itk::MeanSquaresImageToImageMetric< 
                                    FixedImageType, 
                                    MovingImageType >    MetricType;
  typedef itk:: LinearInterpolateImageFunction< 
                                    MovingImageType,
                                    double          >    InterpolatorType;
  typedef itk::ImageRegistrationMethod< 
                                    FixedImageType, 
                                    MovingImageType >    RegistrationType;

  MetricType::Pointer         metric        = MetricType::New();
  OptimizerType::Pointer      optimizer     = OptimizerType::New();
  InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
  RegistrationType::Pointer   registration  = RegistrationType::New();
  
  registration->SetMetric(        metric        );
  registration->SetOptimizer(     optimizer     );
  registration->SetInterpolator(  interpolator  );

  // Se construye la transformación y se conecta al método de registro
  TransformType::Pointer  transform = TransformType::New();
  registration->SetTransform( transform );

  // Lectura de las imágenes de entrada y conexión al método de registro
  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;

  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(  argv[1] );
  movingImageReader->SetFileName( argv[2] );

  registration->SetFixedImage(    fixedImageReader->GetOutput()    );
  registration->SetMovingImage(   movingImageReader->GetOutput()   );
  fixedImageReader->Update();

  registration->SetFixedImageRegion( 
     fixedImageReader->GetOutput()->GetBufferedRegion() );

  // Se actualizan los lectores para garantizar que los parámetros de las
  // imágenes son válidos al momento de inicializar la transformación
  fixedImageReader->Update();
  movingImageReader->Update();

  typedef FixedImageType::SpacingType    SpacingType;
  typedef FixedImageType::PointType      OriginType;
  typedef FixedImageType::RegionType     RegionType;
  typedef FixedImageType::SizeType       SizeType;

  // Como es una transformación centrada, se calcula el centro de la imagen fija
  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();

  const SpacingType fixedSpacing = fixedImage->GetSpacing();
  const OriginType  fixedOrigin  = fixedImage->GetOrigin();
  const RegionType  fixedRegion  = fixedImage->GetLargestPossibleRegion(); 
  const SizeType    fixedSize    = fixedRegion.GetSize();
  
  TransformType::InputPointType centerFixed;
  
  centerFixed[0] = fixedOrigin[0] + fixedSpacing[0] * fixedSize[0] / 2.0;
  centerFixed[1] = fixedOrigin[1] + fixedSpacing[1] * fixedSize[1] / 2.0;

  // Y de manera similar se calcula el centro de la imagen flotante
  MovingImageType::Pointer movingImage = movingImageReader->GetOutput();

  const SpacingType movingSpacing = movingImage->GetSpacing();
  const OriginType  movingOrigin  = movingImage->GetOrigin();
  const RegionType  movingRegion  = movingImage->GetLargestPossibleRegion();
  const SizeType    movingSize    = movingRegion.GetSize();
  
  TransformType::InputPointType centerMoving;
  
  centerMoving[0] = movingOrigin[0] + movingSpacing[0] * movingSize[0] / 2.0;
  centerMoving[1] = movingOrigin[1] + movingSpacing[1] * movingSize[1] / 2.0;

  // La transformación se inicializa en este caso fijando su centro como el de la 
  // imagen fija, y fijando la traslación como la diferencia entre los centros de 
  // las imágenes
  transform->SetCenter( centerFixed );
  transform->SetTranslation( centerMoving - centerFixed );

  // La rotación se inicializa con un ángulo de cero grados
  transform->SetAngle( 0.0 );

  // Los parámetros actuales, que acabamos de fijar, se utilizan para
  // inicializar la transformación como se hacía en los anteriores ejemplos
  registration->SetInitialTransformParameters( transform->GetParameters() );

  // El optimizador provee una funcionalidad de reescalamiento, teniendo en cuenta
  // que las unidades de rotación y traslación son diferentes
  typedef OptimizerType::ScalesType       OptimizerScalesType;
  OptimizerScalesType optimizerScales( transform->GetNumberOfParameters() );
  const double translationScale = 1.0 / 1000.0;

  optimizerScales[0] = 1.0;
  optimizerScales[1] = translationScale;
  optimizerScales[2] = translationScale;
  optimizerScales[3] = translationScale;
  optimizerScales[4] = translationScale;

  optimizer->SetScales( optimizerScales );

  // A continuación hay que fijar los parámetros del método de optimización
  double initialStepLength = 0.1;

  if( argc > 6 )
    {
    initialStepLength = atof( argv[6] );
    }

  optimizer->SetRelaxationFactor( 0.6 );
  optimizer->SetMaximumStepLength( initialStepLength ); 
  optimizer->SetMinimumStepLength( 0.001 );
  optimizer->SetNumberOfIterations( 200 );

  // Crear el comando/observador y registrarlo con el optimizador
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );
 
  // Ejecutar el método de registro
  try 
    { 
    registration->Update(); 
    std::cout << "Optimizer stop condition: "
              << registration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
    } 
  catch( itk::ExceptionObject & err ) 
    { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
    } 
  
  // Obtener los parámetros finales del registro
  OptimizerType::ParametersType finalParameters = 
                    registration->GetLastTransformParameters();

  const double finalAngle           = finalParameters[0];
  const double finalRotationCenterX = finalParameters[1];
  const double finalRotationCenterY = finalParameters[2];
  const double finalTranslationX    = finalParameters[3];
  const double finalTranslationY    = finalParameters[4];

  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();

  const double bestValue = optimizer->GetValue();

  // Imprimir los resultados
  const double finalAngleInDegrees = finalAngle * 180.0 / vnl_math::pi;

  std::cout << "Result = " << std::endl;
  std::cout << " Angle (radians)   = " << finalAngle  << std::endl;
  std::cout << " Angle (degrees)   = " << finalAngleInDegrees  << std::endl;
  std::cout << " Center X      = " << finalRotationCenterX  << std::endl;
  std::cout << " Center Y      = " << finalRotationCenterY  << std::endl;
  std::cout << " Translation X = " << finalTranslationX  << std::endl;
  std::cout << " Translation Y = " << finalTranslationY  << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue          << std::endl;

  // Aplicar la transformación obtenida a la imagen flotante
  typedef itk::ResampleImageFilter< 
                            MovingImageType, 
                            FixedImageType >    ResampleFilterType;

  TransformType::Pointer finalTransform = TransformType::New();

  finalTransform->SetParameters( finalParameters );
  finalTransform->SetFixedParameters( transform->GetFixedParameters() );

  ResampleFilterType::Pointer resample = ResampleFilterType::New();

  resample->SetTransform( finalTransform );
  resample->SetInput( movingImageReader->GetOutput() );

  resample->SetSize(    fixedImage->GetLargestPossibleRegion().GetSize() );
  resample->SetOutputOrigin(  fixedImage->GetOrigin() );
  resample->SetOutputSpacing( fixedImage->GetSpacing() );
  resample->SetOutputDirection( fixedImage->GetDirection() );
  resample->SetDefaultPixelValue( 100 );
  
  // Definir el filtro de escritura de la imagen transformada
  typedef itk::ImageFileWriter< FixedImageType >  WriterFixedType;

  WriterFixedType::Pointer      writer =  WriterFixedType::New();

  writer->SetFileName( argv[3] );

  writer->SetInput( resample->GetOutput()   );

  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    { 
    std::cerr << "ExceptionObject while writing the resampled image !" << std::endl; 
    std::cerr << excp << std::endl; 
    return EXIT_FAILURE;
    } 

  // Generar la imagen diferencia utilizando el filtro de sustracción
  typedef itk::Image< float, Dimension > DifferenceImageType;

  typedef itk::SubtractImageFilter< 
                           FixedImageType, 
                           FixedImageType, 
                           DifferenceImageType > DifferenceFilterType;

  DifferenceFilterType::Pointer difference = DifferenceFilterType::New();

  typedef  unsigned char  OutputPixelType;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  
  typedef itk::RescaleIntensityImageFilter< 
                                  DifferenceImageType, 
                                  OutputImageType >   RescalerType;

  RescalerType::Pointer intensityRescaler = RescalerType::New();

  intensityRescaler->SetOutputMinimum(   0 );
  intensityRescaler->SetOutputMaximum( 255 );

  difference->SetInput1( fixedImageReader->GetOutput() );
  difference->SetInput2( resample->GetOutput() );

  resample->SetDefaultPixelValue( 1 );

  intensityRescaler->SetInput( difference->GetOutput() );  
  
  // Definir el filtro de escritura de la imagen diferencia
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  WriterType::Pointer      writer2 =  WriterType::New();

  writer2->SetInput( intensityRescaler->GetOutput() );


  try
    {
    // Escribir la imagen diferencia después del registro
    if( argc > 4 )
      {
      writer2->SetFileName( argv[4] );
      writer2->Update();
      }

    // Calcular y escribir la imagen diferencia antes del registro
    TransformType::Pointer identityTransform = TransformType::New();
    identityTransform->SetIdentity();
    resample->SetTransform( identityTransform );
    if( argc > 5 )
      {
      writer2->SetFileName( argv[5] );
      writer2->Update();
      }
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << "Error while writing difference images" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
  
}
