import itk
import sys

# sample usage
# ./connectedThreshold input output 100 170 132 142 96

if len(sys.argv) < 8 :
  print("Usage: ", sys.argv[0], " <InputImage> <OutputImage> <LowerThreshold> <UpperThreshold> <XSeed> <YSeed> <ZSeed>")
  sys.exit()

Dimension = 3
PixelType = itk.US

InputImage = sys.argv[1]
OutputImage = sys.argv[2]

LowerThreshold = int(sys.argv[3])
UpperThreshold = int(sys.argv[4])

XSeed = int(sys.argv[5])
YSeed = int(sys.argv[6])
ZSeed = int(sys.argv[7])

ImageType = itk.Image[PixelType, Dimension]

ReaderType = itk.ImageFileReader[ImageType]
reader = ReaderType.New()
reader.SetFileName(InputImage)
reader.Update()

image = reader.GetOutput()
region = image.GetLargestPossibleRegion()
size = region.GetSize()
print(size)

FilterType = itk.ConnectedThresholdImageFilter[ImageType, ImageType]
connectedThreshold = FilterType.New()
connectedThreshold.SetLower(LowerThreshold)
connectedThreshold.SetUpper(UpperThreshold)
connectedThreshold.SetReplaceValue(255)

seed = []
seed.append(XSeed)
seed.append(YSeed)
seed.append(ZSeed)
connectedThreshold.SetSeed(seed)
connectedThreshold.SetInput(reader.GetOutput())

RescaleType = itk.RescaleIntensityImageFilter[ImageType, ImageType]
rescaler = RescaleType.New()
rescaler.SetInput(connectedThreshold.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

WriterType = itk.ImageFileWriter[ImageType]
writer = WriterType.New()
writer.SetFileName(OutputImage)
writer.SetInput(rescaler.GetOutput())
writer.Update()
