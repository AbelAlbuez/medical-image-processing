#!/usr/bin/env python

import itk
import argparse

parser = argparse.ArgumentParser(description="Gradient Magnitude Filtering Of An Image.")
parser.add_argument("input_image")
parser.add_argument("output_image")
args = parser.parse_args()

InputPixelType = itk.UC
OutputPixelType = itk.F
Dimension = 3

InputImageType = itk.Image[InputPixelType, Dimension]
OutputImageType = itk.Image[OutputPixelType, Dimension]

reader = itk.ImageFileReader[InputImageType].New()
reader.SetFileName(args.input_image)

gradientFilter = itk.GradientMagnitudeImageFilter[InputImageType, OutputImageType].New()
gradientFilter.SetInput(reader.GetOutput())

writer = itk.ImageFileWriter[OutputImageType].New()
writer.SetFileName(args.output_image)
writer.SetInput(gradientFilter.GetOutput())

writer.Update()
