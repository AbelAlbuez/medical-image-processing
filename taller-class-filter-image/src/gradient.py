#!/usr/bin/env python
"""
Taller: GradientMagnitudeImageFilter — magnitud del gradiente (3D).
Nota ITK Python: entrada y salida deben usar un par de tipos soportados (p. ej. UC→UC).
"""

import argparse
import itk

parser = argparse.ArgumentParser(description="Gradient Magnitude Filtering (taller).")
parser.add_argument("input_image")
parser.add_argument("output_image")
args = parser.parse_args()

PixelType = itk.UC
Dimension = 3
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.input_image)

# --- TALLER: GradientMagnitudeImageFilter[ImageType, ImageType] (mismo tipo entrada/salida) ---
# gradientFilter = itk.GradientMagnitudeImageFilter[ImageType, ImageType].New()
# gradientFilter.SetInput(reader.GetOutput())

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_image)
# writer.SetInput(gradientFilter.GetOutput())
writer.SetInput(reader.GetOutput())

writer.Update()
