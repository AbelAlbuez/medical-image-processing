#include "itkImageFileReader.h"
#include "itkConfidenceConnectedImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageFileWriter.h"

int
main(int argc, char * argv[])
{
  // sample usage
  //./confidenceConnected input output 0 2 1 132 142 96

  if (argc < 9)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << std::endl;
    std::cerr << " <InputImage> <OutputImage> <NumberOfIterations> <Multiplier>";
    std::cerr << " <InitialNeighborhoodRadius> <XSeed> <YSeed> <ZSeed>";
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using PixelType = short;
  using SizeType = itk::SizeValueType;

  const char * InputImage = argv[1];
  const char * OutputImage = argv[2];

  const auto NumberOfIterations = static_cast<SizeType>(atoi(argv[3]));
  const auto Multiplier = static_cast<SizeType>(atoi(argv[4]));
  const auto InitialNeighborhoodRadius = static_cast<SizeType>(atoi(argv[5]));

  const auto XSeed = static_cast<SizeType>(atoi(argv[6]));
  const auto YSeed = static_cast<SizeType>(atoi(argv[7]));
  const auto ZSeed = static_cast<SizeType>(atoi(argv[8]));

  using ImageType = itk::Image<PixelType, Dimension>;

  using ReaderType = itk::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(InputImage);
  reader->Update();

  ImageType::Pointer image = reader->GetOutput();
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();
  std::cout << size << std::endl;

  using FilterType = itk::ConfidenceConnectedImageFilter<ImageType, ImageType>;
  FilterType::Pointer confidenceConnected = FilterType::New();
  confidenceConnected->SetInitialNeighborhoodRadius(InitialNeighborhoodRadius);
  confidenceConnected->SetMultiplier(Multiplier);
  confidenceConnected->SetNumberOfIterations(NumberOfIterations);
  confidenceConnected->SetReplaceValue(255);

  ImageType::IndexType seed;
  seed[0] = XSeed;
  seed[1] = YSeed;
  seed[2] = ZSeed;
  confidenceConnected->SetSeed(seed);
  confidenceConnected->SetInput(reader->GetOutput());

  using RescaleType = itk::RescaleIntensityImageFilter<ImageType, ImageType>;
  RescaleType::Pointer rescaler = RescaleType::New();
  rescaler->SetInput(confidenceConnected->GetOutput());
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