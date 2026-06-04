"""
limpieza_core.py
================
Núcleo numérico de la limpieza de T1c. Copia/adapta las "joyas" del paquete de Santiago
(denoise.py, bias_field.py, normalize.py), reorquestadas para SOLO T1c y con la
normalización por percentiles como esquema por defecto del proyecto.

Orden (del enunciado):  denoise (Wiener) -> N4 (sesgo) -> normalización percentil [0,1].
  1. Denoise primero: reduce el ruido que, si no, N4 podría modelar como estructura.
  2. N4 corrige el sesgo multiplicativo de campo.
  3. Normalización al final fija la escala (la escala cruda varía mucho entre casos, EDA).

Motor: SimpleITK (N4 es exclusivo de ITK/SimpleITK). Las funciones devuelven imágenes
SITK con la geometría preservada.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.signal import wiener
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import io_zip   # noqa: E402


# --------------------------------------------------------------------------- #
# 1) Denoise — Adaptive Wiener Filter (copiado de denoise.py de Santiago)
# --------------------------------------------------------------------------- #
def adaptive_wiener(volumen: np.ndarray, mysize: int = 3,
                    noise: Optional[float] = None) -> np.ndarray:
    """
    Adaptive Wiener Filter 3D. En zonas homogéneas suaviza fuerte; en bordes preserva
    detalle. Repone el fondo (skull-stripped) a 0 y limpia NaN de varianza local nula.
    """
    v = np.asarray(volumen, dtype=np.float64)
    mascara = v > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        filtrado = wiener(v, mysize=mysize, noise=noise)
    filtrado = np.nan_to_num(filtrado, nan=0.0, posinf=0.0, neginf=0.0)
    filtrado[~mascara] = 0.0
    return filtrado.astype(np.float32)


def denoise_sitk(img: sitk.Image, mysize: int = 3) -> sitk.Image:
    """Wrapper SITK->SITK del Adaptive Wiener Filter (preserva geometría)."""
    arr = io_zip.a_numpy(img)
    out = adaptive_wiener(arr, mysize=mysize)
    return io_zip.desde_numpy(out, ref=img)


# --------------------------------------------------------------------------- #
# 2) N4 — corrección de campo de sesgo (copiado de bias_field.py de Santiago)
# --------------------------------------------------------------------------- #
def corregir_n4(img: sitk.Image, mascara: Optional[sitk.Image] = None,
                shrink_factor: int = 4, iteraciones: Optional[List[int]] = None,
                fwhm: float = 0.15) -> sitk.Image:
    """
    Aplica N4 a T1c. Estima el campo en baja resolución (shrink) y lo reconstruye a
    resolución completa con GetLogBiasFieldAsImage. Repone el fondo a 0 fuera de la máscara.
    """
    if mascara is None:
        mascara = io_zip.mascara_cerebro(img)
    if iteraciones is None:
        iteraciones = [50, 50, 50, 50]

    img_s = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    msk_s = sitk.Shrink(mascara, [shrink_factor] * mascara.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iteraciones)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(fwhm)
    _ = corrector.Execute(img_s, msk_s)

    log_bias = corrector.GetLogBiasFieldAsImage(img)
    corregida = img / sitk.Exp(log_bias)
    corregida = sitk.Mask(corregida, mascara)
    return sitk.Cast(corregida, sitk.sitkFloat32)


# --------------------------------------------------------------------------- #
# 3) Normalización por percentiles (de normalize.py de Santiago; default percentil)
# --------------------------------------------------------------------------- #
def percentil_a_unidad(volumen: np.ndarray, p_lo: float = 0.5,
                       p_hi: float = 99.5) -> np.ndarray:
    """Recorta a [p_lo, p_hi] (percentiles dentro del cerebro) y escala a [0,1]."""
    v = np.asarray(volumen, dtype=np.float32)
    m = v > 0
    if m.sum() == 0:
        return v
    lo, hi = np.percentile(v[m], [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1.0
    out = np.zeros_like(v)
    out[m] = np.clip((v[m] - lo) / (hi - lo), 0.0, 1.0)
    return out


def normalizar_percentil_sitk(img: sitk.Image, p_lo: float = 0.5,
                              p_hi: float = 99.5) -> sitk.Image:
    """Wrapper SITK->SITK de la normalización por percentiles a [0,1]."""
    arr = io_zip.a_numpy(img)
    out = percentil_a_unidad(arr, p_lo, p_hi)
    return io_zip.desde_numpy(out.astype(np.float32), ref=img)


# --------------------------------------------------------------------------- #
# Pipeline completo de un volumen T1c
# --------------------------------------------------------------------------- #
def limpiar_t1c(img: sitk.Image, mysize_wiener: int = 3, n4_shrink: int = 4,
                p_lo: float = 0.5, p_hi: float = 99.5) -> sitk.Image:
    """denoise (Wiener) -> N4 -> normalización percentil [0,1]. Devuelve imagen SITK."""
    mascara = io_zip.mascara_cerebro(img)
    img = denoise_sitk(img, mysize=mysize_wiener)
    img = corregir_n4(img, mascara=mascara, shrink_factor=n4_shrink)
    img = normalizar_percentil_sitk(img, p_lo=p_lo, p_hi=p_hi)
    return img
