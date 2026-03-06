#!/usr/bin/env python

from tokenize import Double3
from unicodedata import decimal
import itk
import argparse

parser = argparse.ArgumentParser(description="Mean Filtering Of An Image.")
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("radius", type=int)
args = parser.parse_args()

PixelType = itk.UC
Dimension = 3

ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.input_image)

meanFilter = itk.MeanImageFilter[ImageType, ImageType].New()
meanFilter.SetInput(reader.GetOutput())
meanFilter.SetRadius(args.radius)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_image)
writer.SetInput(meanFilter.GetOutput())

writer.Update()