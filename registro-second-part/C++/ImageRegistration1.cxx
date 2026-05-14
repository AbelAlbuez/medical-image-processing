/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: ImageRegistration1.cxx,v $
  Language:  C++
  Date:      $Date: 2007/11/22 00:30:16 $
  Version:   $Revision: 1.53 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "itkImageRegistrationMethod.h"
#include "itkTranslationTransform.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkImage.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"


// para hacer seguimiento al proceso de optimización, ITK utiliza un esquema
// de comando/observador, que permite tener información iteración a iteración
class CommandIterationUpdate : public itk::Command 
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate() {};

public:

  typedef itk::RegularStepGradientDescentOptimizer     OptimizerType;
  typedef const OptimizerType                         *OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event) override
  {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    OptimizerPointer optimizer = 
                         dynamic_cast< OptimizerPointer >( object );

    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
      return;
      }

    std::cout << optimizer->GetCurrentIteration() << " = ";
    std::cout << optimizer->GetValue() << " : ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;
  }
   
};


int main( int argc, char *argv[] )
{
  //Imágenes de entrada:
  // Imagen fija: BrainProtonDensitySliceBorder20.png
  // Imagen flotante: BrainProtonDensitySliceShifted13x17y.png

  // identificar la cantidad de parámetros de entrada del comando
  // si no son suficientes, mostrar el uso del programa
  if( argc < 4 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile  movingImageFile ";
    std::cerr << "outputImagefile [differenceImageAfter]";
    std::cerr << "[differenceImageBefore]" << std::endl;
    return EXIT_FAILURE;
    }

  // Identificar la dimensión de la imagen y el tipo de dato a usar
  const    unsigned int    Dimension = 2;
  typedef  float           PixelType;

  // Definición de los tipos de las imágenes de entrada
  typedef itk::Image< PixelType, Dimension >  FixedImageType;
  typedef itk::Image< PixelType, Dimension >  MovingImageType;

  // Definición de la transformación que lleva una imagen flotante a la imagen fija
  typedef itk::TranslationTransform< double, Dimension > TransformType;

  // Definición del optimizador que explora el espacio de parámetros de la
  // transformación en busca de los valores óptimos de la métrica
  typedef itk::RegularStepGradientDescentOptimizer       OptimizerType;

  // Definición de la métrica que identifica la similaridad entre las imágenes
  typedef itk::MeanSquaresImageToImageMetric< 
                                    FixedImageType, 
                                    MovingImageType >    MetricType;

  // Definición del interpolador, que evalúa las intensidades de la imagen
  // flotante en posiciones no ajustadas a la grilla
  typedef itk:: LinearInterpolateImageFunction< 
                                    MovingImageType,
                                    double          >    InterpolatorType;

  // Definición del método de registro, que interconecta los elementos
  // definidos hasta ahora
  typedef itk::ImageRegistrationMethod< 
                                    FixedImageType, 
                                    MovingImageType >    RegistrationType;

  // Cada uno de los componentes del método de registro se crea utilizando
  // su método New() y se asigna a su apuntador (SmartPointer) respectivo
  MetricType::Pointer         metric        = MetricType::New();
  TransformType::Pointer      transform     = TransformType::New();
  OptimizerType::Pointer      optimizer     = OptimizerType::New();
  InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
  RegistrationType::Pointer   registration  = RegistrationType::New();

  // Cada componente se conecta con el método de registro
  registration->SetMetric(        metric        );
  registration->SetOptimizer(     optimizer     );
  registration->SetTransform(     transform     );
  registration->SetInterpolator(  interpolator  );

  // Las imágenes de entrada (fija y flotante) se leen de archivos 
  // externos utilizando clases lectoras (readers), utilizando los 
  // parámetros del programa como los nombres de archivo
  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;
  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(  argv[1] );
  movingImageReader->SetFileName( argv[2] );

  // Las imágenes se asignan al método de registro
  registration->SetFixedImage(    fixedImageReader->GetOutput()    );
  registration->SetMovingImage(   movingImageReader->GetOutput()   );

  // El registro puede realizarse sólo en una parte de la imagen, en este
  // caso se usa el contenido disponible de la imagen como región de registro
  fixedImageReader->Update();
  registration->SetFixedImageRegion( 
                    fixedImageReader->GetOutput()->GetBufferedRegion() );

  // Definición de los parámetros iniciales de la transformación
  // en este caso se define una traslación en ceros (transformación identidad)
  typedef RegistrationType::ParametersType ParametersType;
  ParametersType initialParameters( transform->GetNumberOfParameters() );

  initialParameters[0] = 0.0;  // Initial offset in mm along X
  initialParameters[1] = 0.0;  // Initial offset in mm along Y
  
  registration->SetInitialTransformParameters( initialParameters );

  // Definición de algunos parámetros del optimizador
  optimizer->SetMaximumStepLength( 4.00 );  
  optimizer->SetMinimumStepLength( 0.01 );
  optimizer->SetNumberOfIterations( 200 );

  // Se fija el comando/observador, para seguir el progreso del registro
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  // El proceso de registro se ejecuta haciendo un llamado al método Update()
  // del filtro. Si algo anda mal, se genera una excepción de error.
  try 
    { 
    registration->Update(); 
    } 
  catch( itk::ExceptionObject & err ) 
    { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
    } 

  // El resultado del proceso de registro es un conjunto de parámetros que definen
  // de manera única la transformación. El optimizador puede retornar también
  // la cantidad de iteraciones realizadas y el valor de la métrica.
  ParametersType finalParameters = registration->GetLastTransformParameters();

  const double TranslationAlongX = finalParameters[0];
  const double TranslationAlongY = finalParameters[1];

  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
  const double bestValue = optimizer->GetValue();

  std::cout << "Result = " << std::endl;
  std::cout << " Translation X = " << TranslationAlongX  << std::endl;
  std::cout << " Translation Y = " << TranslationAlongY  << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue          << std::endl;

  // Hasta aquí, sólo se tienen los parámetros de la transformación.
  // Usualmente, se aplica la transformación a la imagen flotante para verla transformada
  // Esto se realiza a través de un filtro de remuestreo de la imagen
  typedef itk::ResampleImageFilter< 
                            MovingImageType, 
                            FixedImageType >    ResampleFilterType;
  
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetInput( movingImageReader->GetOutput() );
  resampler->SetTransform( registration->GetOutput()->Get() );
  
  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
  resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  resampler->SetOutputOrigin(  fixedImage->GetOrigin() );
  resampler->SetOutputSpacing( fixedImage->GetSpacing() );
  resampler->SetOutputDirection( fixedImage->GetDirection() );
  resampler->SetDefaultPixelValue( 100 );
  
  // La salida del filtro de remuestreo se pasa a un filtro de escritura que genera la
  // imagen en disco, utilizando un conversor de tipo de dato
  typedef unsigned char OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::CastImageFilter< 
                        FixedImageType,
                        OutputImageType > CastFilterType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  
  WriterType::Pointer      writer =  WriterType::New();
  CastFilterType::Pointer  caster =  CastFilterType::New();
  
  writer->SetFileName( argv[3] );

  caster->SetInput( resampler->GetOutput() );
  writer->SetInput( caster->GetOutput()   );
  writer->Update();

  // La imagen fija y la flotante (transformada) pueden compararse entre sí por medio
  // de un filtro diferencia (sustracción)
  typedef itk::SubtractImageFilter< 
                                  FixedImageType, 
                                  FixedImageType, 
                                  FixedImageType > DifferenceFilterType;

  DifferenceFilterType::Pointer difference = DifferenceFilterType::New();

  difference->SetInput1( fixedImageReader->GetOutput() );
  difference->SetInput2( resampler->GetOutput() );

  // Las diferencias pueden ser muy pequeñas para ser visualizadas, por lo que se
  // pasan a través de un filtro que reescala los rangos de intensidades
  typedef itk::RescaleIntensityImageFilter< 
                                  FixedImageType, 
                                  OutputImageType >   RescalerType;

  RescalerType::Pointer intensityRescaler = RescalerType::New();
  
  intensityRescaler->SetInput( difference->GetOutput() );
  intensityRescaler->SetOutputMinimum(   0 );
  intensityRescaler->SetOutputMaximum( 255 );

  resampler->SetDefaultPixelValue( 1 );

  // esta diferencia puede escribirse a un archivo
  WriterType::Pointer writer2 = WriterType::New();
  writer2->SetInput( intensityRescaler->GetOutput() );  

  if( argc > 4 )
    {
    writer2->SetFileName( argv[4] );
    writer2->Update();
    }

  // Las diferencias antes de la transformación pueden establecerse aplicando el filtro
  // de diferencias a la imagen flotante con una transformación identidad 
  TransformType::Pointer identityTransform = TransformType::New();
  identityTransform->SetIdentity();
  resampler->SetTransform( identityTransform );

  if( argc > 5 )
    {
    writer2->SetFileName( argv[5] );
    writer2->Update();
    }

  return EXIT_SUCCESS;
}

