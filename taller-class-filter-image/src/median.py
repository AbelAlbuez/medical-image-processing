#!/usr/bin/env python
"""
Taller: MedianImageFilter — filtro de mediana (3D).
"""

import argparse
import itk

parser = argparse.ArgumentParser(description="Median Filtering Of An Image (taller).")
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("radius", type=int)
args = parser.parse_args()

PixelType = itk.UC
Dimension = 3
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.input_image)

# --- TALLER: MedianImageFilter, SetRadius(args.radius) ---
# medianFilter = itk.MedianImageFilter[ImageType, ImageType].New()
# medianFilter.SetInput(reader.GetOutput())
# medianFilter.SetRadius(args.radius)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_image)
# writer.SetInput(medianFilter.GetOutput())
writer.SetInput(reader.GetOutput())

writer.Update()
