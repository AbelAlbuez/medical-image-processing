"""
bias_field.py
=============
Corrección del campo de sesgo (intensity inhomogeneity) con **N4** de SimpleITK.

El sesgo de campo es una modulación suave y multiplicativa de la intensidad
causada por la bobina de RF; hace que el mismo tejido se vea más brillante en una
región que en otra. N4 estima ese campo log-aditivo y lo divide.

Buenas prácticas aplicadas:
  * Estimar/aplicar dentro de la máscara de cerebro (evita ajustar al fondo).
  * `shrinkFactor` para acelerar (estima el campo en baja resolución y luego lo
    reconstruye a resolución completa con `GetLogBiasFieldAsImage`).
  * Número de iteraciones por nivel configurable.

N4 es exclusivo de ITK/SimpleITK; este módulo no tiene equivalente en numpy puro.
"""
from __future__ import annotations
from typing import List, Optional

import SimpleITK as sitk
from . import io_utils


def corregir_n4(img: sitk.Image,
                mascara: Optional[sitk.Image] = None,
                shrink_factor: int = 4,
                iteraciones: Optional[List[int]] = None,
                fwhm: float = 0.15) -> sitk.Image:
    """
    Aplica N4 a una modalidad.

    Parameters
    ----------
    img : sitk.Image (float32)
    mascara : sitk.Image uint8, opcional
        Máscara de cerebro. Si None se usa `img > 0` (BraTS skull-stripped).
    shrink_factor : int
        Submuestreo para estimar el campo (4 es buen compromiso velocidad/calidad;
        usa 2 para máxima calidad si tienes tiempo).
    iteraciones : lista de int
        Iteraciones por nivel multiresolución. Default [50, 50, 50, 50].
    fwhm : float
        Ancho del histograma del modelo (biasFieldFullWidthAtHalfMaximum).

    Returns
    -------
    sitk.Image corregida, a resolución COMPLETA.
    """
    if mascara is None:
        mascara = io_utils.mascara_cerebro(img)
    if iteraciones is None:
        iteraciones = [50, 50, 50, 50]

    # Estimación en baja resolución (rápida).
    img_s = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    msk_s = sitk.Shrink(mascara, [shrink_factor] * mascara.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iteraciones)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(fwhm)
    _ = corrector.Execute(img_s, msk_s)

    # Reconstruir el campo a resolución completa y aplicarlo al volumen original
    # (mejor que devolver la salida submuestreada del corrector).
    log_bias = corrector.GetLogBiasFieldAsImage(img)
    corregida = img / sitk.Exp(log_bias)
    # Reponer el fondo a 0 fuera de la máscara.
    corregida = sitk.Mask(corregida, mascara)
    return sitk.Cast(corregida, sitk.sitkFloat32)
