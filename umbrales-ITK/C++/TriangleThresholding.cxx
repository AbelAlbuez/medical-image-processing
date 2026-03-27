#include "itkImageFileReader.h"
#include "itkTriangleThresholdImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageFileWriter.h"

int
main(int argc, char * argv[])
{
  // sample usage
  //./triangleClustering input output 128

  if (argc < 4)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << std::endl;
    std::cerr << " <InputImage> <OutputImage> <NumberOfBins>";
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using PixelType = short;
  using SizeType = itk::SizeValueType;

  const char * InputImage = argv[1];
  const char * OutputImage = argv[2];

  const auto NumberOfHistogramBins = static_cast<SizeType>(atoi(argv[3]));

  using ImageType = itk::Image<PixelType, Dimension>;

  using ReaderType = itk::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(InputImage);

  using FilterType = itk::TriangleThresholdImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(reader->GetOutput());
  filter->SetNumberOfHistogramBins(NumberOfHistogramBins);

  using RescaleType = itk::RescaleIntensityImageFilter<ImageType, ImageType>;
  RescaleType::Pointer rescaler = RescaleType::New();
  rescaler->SetInput(filter->GetOutput());
  rescaler->SetOutputMinimum(0);
  rescaler->SetOutputMaximum(255);

  using WriterType = itk::ImageFileWriter<ImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(OutputImage);
  writer->SetInput(rescaler->GetOutput());

  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject & e)
  {
    std::cerr << "Error: " << e << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}