import itk
import sys

# sample usage
# ./confidenceConnected input output 0 2 1 132 142 96

# estos parametros son la semilla => 132 142 96 (un lado del ventriculo)



if len(sys.argv) < 9 :
  print("Usage: ", sys.argv[0], " <InputImage> <OutputImage> <NumberOfIterations> <Multiplier> <InitialNeighborhoodRadius> <XSeed> <YSeed> <ZSeed>")
  sys.exit()

Dimension = 3
PixelType = itk.US

InputImage = sys.argv[1]
OutputImage = sys.argv[2]

NumberOfIterations = int(sys.argv[3])
Multiplier = int(sys.argv[4])
InitialNeighborhoodRadius = int(sys.argv[5])

XSeed = int(sys.argv[6])
YSeed = int(sys.argv[7])
ZSeed = int(sys.argv[8])

ImageType = itk.Image[PixelType, Dimension]

ReaderType = itk.ImageFileReader[ImageType]
reader = ReaderType.New()
reader.SetFileName(InputImage)
reader.Update()

image = reader.GetOutput()
region = image.GetLargestPossibleRegion()
size = region.GetSize()
print(size)

FilterType = itk.ConfidenceConnectedImageFilter[ImageType, ImageType]
confidenceConnected = FilterType.New()
confidenceConnected.SetInitialNeighborhoodRadius(InitialNeighborhoodRadius)
confidenceConnected.SetMultiplier(Multiplier)
confidenceConnected.SetNumberOfIterations(NumberOfIterations)
confidenceConnected.SetReplaceValue(255)

seed = []
seed.append(XSeed)
seed.append(YSeed)
seed.append(ZSeed)
confidenceConnected.SetSeed(seed)
confidenceConnected.SetInput(reader.GetOutput())

RescaleType = itk.RescaleIntensityImageFilter[ImageType, ImageType]
rescaler = RescaleType.New()
rescaler.SetInput(confidenceConnected.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

WriterType = itk.ImageFileWriter[ImageType]
writer = WriterType.New()
writer.SetFileName(OutputImage)
writer.SetInput(rescaler.GetOutput())
writer.Update()
