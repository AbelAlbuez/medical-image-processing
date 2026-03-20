#!/usr/bin/env python
"""
Taller: AdaptiveHistogramEqualizationImageFilter (3D).
"""

import argparse
import itk

parser = argparse.ArgumentParser(
    description="Adaptive Histogram Equalization Image Filter (taller)."
)
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("alpha", type=float)
parser.add_argument("beta", type=float)
parser.add_argument("radius", type=int)
args = parser.parse_args()

Dimension = 3
PixelType = itk.ctype("unsigned char")
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.input_image)

# --- TALLER: AdaptiveHistogramEqualizationImageFilter.New(reader), SetAlpha, SetBeta, SetRadius ---
# histogramEqualization = itk.AdaptiveHistogramEqualizationImageFilter.New(reader)
# histogramEqualization.SetAlpha(args.alpha)
# histogramEqualization.SetBeta(args.beta)
# radius = itk.Size[Dimension]()
# radius.Fill(args.radius)
# histogramEqualization.SetRadius(radius)
# itk.imwrite(histogramEqualization, args.output_image)
# sys.exit(0)

# Temporal: copia sin equalización hasta implementar arriba
writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_image)
writer.SetInput(reader.GetOutput())
writer.Update()
