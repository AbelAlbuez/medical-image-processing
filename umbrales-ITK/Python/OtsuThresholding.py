import itk
import sys

# sample usage
# ./otsuClustering input output 128

if len(sys.argv) != 4 :
  print("Usage: ", sys.argv[0], " <InputImage> <OutputImage> <NumberOfBins>")
  sys.exit()

Dimension = 3
PixelType = itk.US

InputImage = sys.argv[1]
OutputImage = sys.argv[2]
NumberOfHistogramBins = int(sys.argv[3])

ImageType = itk.Image[PixelType, Dimension]

ReaderType = itk.ImageFileReader[ImageType]
reader = ReaderType.New()
reader.SetFileName(InputImage)

FilterType = itk.OtsuThresholdImageFilter[ImageType, ImageType]
imfilter = FilterType.New()
imfilter.SetInput(reader.GetOutput())
imfilter.SetNumberOfHistogramBins(NumberOfHistogramBins)

RescaleType = itk.RescaleIntensityImageFilter[ImageType, ImageType]
rescaler = RescaleType.New()
rescaler.SetInput(imfilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

WriterType = itk.ImageFileWriter[ImageType]
writer = WriterType.New()
writer.SetFileName(OutputImage)
writer.SetInput(rescaler.GetOutput())
writer.Update()
