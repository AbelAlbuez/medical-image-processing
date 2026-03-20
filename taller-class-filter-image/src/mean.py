#!/usr/bin/env python
"""
Taller: MeanImageFilter — suavizado por media (3D).
"""

import argparse
import itk

parser = argparse.ArgumentParser(description="Mean Filtering Of An Image (taller).")
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("radius", type=int)
args = parser.parse_args()

PixelType = itk.UC
Dimension = 3
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.input_image)

# --- TALLER: instanciar MeanImageFilter, SetInput(reader.GetOutput()), SetRadius(args.radius) ---
# meanFilter = itk.MeanImageFilter[ImageType, ImageType].New()
# meanFilter.SetInput(reader.GetOutput())
# meanFilter.SetRadius(args.radius)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_image)
# writer.SetInput(meanFilter.GetOutput())
writer.SetInput(reader.GetOutput())  # temporal: sin filtro hasta implementar arriba

writer.Update()
