#include "itkImageFileReader.h"
#include "itkConnectedThresholdImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageFileWriter.h"

int
main(int argc, char * argv[])
{
  // sample usage
  //./connectedThreshold input output 100 170 132 142 96

  if (argc < 8)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << std::endl;
    std::cerr << " <InputImage> <OutputImage> <LowerThreshold>";
    std::cerr << " <UpperThreshold> <XSeed> <YSeed> <ZSeed>";
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using PixelType = short;
  using SizeType = itk::SizeValueType;

  const char * InputImage = argv[1];
  const char * OutputImage = argv[2];

  const auto LowerThreshold = static_cast<SizeType>(atoi(argv[3]));
  const auto UpperThreshold = static_cast<SizeType>(atoi(argv[4]));

  const auto XSeed = static_cast<SizeType>(atoi(argv[5]));
  const auto YSeed = static_cast<SizeType>(atoi(argv[6]));
  const auto ZSeed = static_cast<SizeType>(atoi(argv[7]));

  using ImageType = itk::Image<PixelType, Dimension>;

  using ReaderType = itk::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(InputImage);
  reader->Update();

  ImageType::Pointer image = reader->GetOutput();
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();
  std::cout << size << std::endl;

  using FilterType = itk::ConnectedThresholdImageFilter<ImageType, ImageType>;
  FilterType::Pointer connectedThreshold = FilterType::New();
  connectedThreshold->SetLower(LowerThreshold);
  connectedThreshold->SetUpper(UpperThreshold);
  connectedThreshold->SetReplaceValue(255);

  ImageType::IndexType seed;
  seed[0] = XSeed;
  seed[1] = YSeed;
  seed[2] = ZSeed;
  connectedThreshold->SetSeed(seed);
  connectedThreshold->SetInput(reader->GetOutput());

  using RescaleType = itk::RescaleIntensityImageFilter<ImageType, ImageType>;
  RescaleType::Pointer rescaler = RescaleType::New();
  rescaler->SetInput(connectedThreshold->GetOutput());
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