#!/usr/bin/env python

import itk
import argparse

parser = argparse.ArgumentParser(
    description="Adaptive Histogram Equalization Image Filter."
)
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("alpha", type=float)
parser.add_argument("beta", type=float)
parser.add_argument("radius", type=int)
args = parser.parse_args()

Dimension = 2

PixelType = itk.ctype("unsigned char")
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.input_image)

histogramEqualization = itk.AdaptiveHistogramEqualizationImageFilter.New(reader)
histogramEqualization.SetAlpha(args.alpha)
histogramEqualization.SetBeta(args.beta)

radius = itk.Size[Dimension]()
radius.Fill(args.radius)
histogramEqualization.SetRadius(radius)

itk.imwrite(histogramEqualization, args.output_image)