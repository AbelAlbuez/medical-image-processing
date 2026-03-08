#!/usr/bin/env python

import itk
import argparse

parser = argparse.ArgumentParser(description="Gradient Magnitude Filtering Of An Image.")
parser.add_argument("input_image")
parser.add_argument("output_image")
args = parser.parse_args()

# ITK Python solo tiene instanciado el filtro con mismo tipo entrada/salida (UC->UC, F->F, etc.)
PixelType = itk.UC
Dimension = 3

ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.input_image)

gradientFilter = itk.GradientMagnitudeImageFilter[ImageType, ImageType].New()
gradientFilter.SetInput(reader.GetOutput())

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_image)
writer.SetInput(gradientFilter.GetOutput())

writer.Update()
