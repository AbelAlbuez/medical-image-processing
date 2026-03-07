#!/usr/bin/env python

import itk
import argparse

parser = argparse.ArgumentParser(description="Median Filtering Of An Image.")
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("radius", type=int)
args = parser.parse_args()

PixelType = itk.UC
Dimension = 3

ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.input_image)

medianFilter = itk.MedianImageFilter[ImageType, ImageType].New()
medianFilter.SetInput(reader.GetOutput())
medianFilter.SetRadius(args.radius)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_image)
writer.SetInput(medianFilter.GetOutput())

writer.Update()