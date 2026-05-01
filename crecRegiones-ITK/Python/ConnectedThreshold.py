import itk
import sys
import numpy as np

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

# --- PRE-PROCESO: enmascarar imagen con ROI alrededor de la semilla ---

# Leer array numpy del volumen original (orden Z, Y, X)
imgArray = itk.array_view_from_image(reader.GetOutput()).copy()

# Definir ROI centrada en la semilla (x=132, y=142, z=96)
# Margen en voxeles por cada eje - ajustar si la estructura queda cortada
ROI_X = (100, 160)   # eje X del volumen -> axis=2 en numpy
ROI_Y = (110, 175)   # eje Y del volumen -> axis=1 en numpy
ROI_Z = (75,  120)   # eje Z del volumen -> axis=0 en numpy

# Crear mascara: todo ceros, solo la ROI conserva valores originales
maskedArray = np.zeros_like(imgArray)
maskedArray[ROI_Z[0]:ROI_Z[1],
            ROI_Y[0]:ROI_Y[1],
            ROI_X[0]:ROI_X[1]] = imgArray[ROI_Z[0]:ROI_Z[1],
                                           ROI_Y[0]:ROI_Y[1],
                                           ROI_X[0]:ROI_X[1]]

# Convertir de vuelta a imagen ITK preservando metadatos del original
maskedImage = itk.image_from_array(maskedArray.astype(np.uint16))
maskedImage.SetSpacing(reader.GetOutput().GetSpacing())
maskedImage.SetOrigin(reader.GetOutput().GetOrigin())
maskedImage.SetDirection(reader.GetOutput().GetDirection())
# --- FIN PRE-PROCESO ---

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
connectedThreshold.SetInput(maskedImage)

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
