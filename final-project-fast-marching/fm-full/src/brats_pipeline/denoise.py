"""
denoise.py
==========
Reducción de ruido con el **Adaptive Wiener Filter (AWF)** — el que quedó
pendiente del Taller 1.

El AWF estima, en una ventana local móvil de tamaño `mysize`, la media (m) y la
varianza (v) de la señal y aplica:

        salida = m + max(0, v - ruido) / max(v, ruido) * (x - m)

Donde `ruido` es la varianza del ruido (si no se da, se estima como el promedio
de las varianzas locales). En zonas homogéneas (v ~ ruido) suaviza fuerte; en
bordes (v >> ruido) preserva el detalle. Es exactamente el filtro adaptativo de
Lee/Wiener, implementado en N-D por `scipy.signal.wiener`, lo que nos da una
versión 3D directa y reproducible.

Trabajamos sobre el ndarray para que la lógica sea testeable sin SimpleITK; el
wrapper `denoise_imagen` mantiene la geometría.
"""
from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal import wiener

import SimpleITK as sitk
from . import io_utils


def adaptive_wiener(volumen: np.ndarray,
                    mysize: Union[int, Tuple[int, int, int]] = 3,
                    noise: Optional[float] = None,
                    solo_cerebro: bool = True) -> np.ndarray:
    """
    Aplica AWF a un volumen 3D.

    Parameters
    ----------
    volumen : ndarray 3D (float)
        Intensidades de una modalidad.
    mysize : int o (a,b,c)
        Tamaño de la ventana local. 3 es un buen punto de partida para MRI 1mm;
        5 suaviza más (úsalo si el ruido es alto). Debe ser impar.
    noise : float, opcional
        Varianza del ruido. Si None, scipy la estima como el promedio de las
        varianzas locales (autoadaptativo).
    solo_cerebro : bool
        Si True, re-impone el fondo en 0 tras filtrar (BraTS es skull-stripped,
        así no "sangra" ruido del borde hacia el fondo).

    Returns
    -------
    ndarray 3D float32 filtrado.
    """
    v = np.asarray(volumen, dtype=np.float64)
    mascara = v > 0
    # En zonas homogéneas la varianza local es 0 -> división interna 0/0 (benigna).
    with np.errstate(divide="ignore", invalid="ignore"):
        filtrado = wiener(v, mysize=mysize, noise=noise)
    # wiener puede devolver NaN donde la varianza local es 0 -> los reponemos.
    filtrado = np.nan_to_num(filtrado, nan=0.0, posinf=0.0, neginf=0.0)
    if solo_cerebro:
        filtrado[~mascara] = 0.0
    return filtrado.astype(np.float32)


def denoise_imagen(img: sitk.Image, mysize: int = 3,
                   noise: Optional[float] = None) -> sitk.Image:
    """Wrapper SITK->SITK que preserva la geometría del volumen."""
    arr = io_utils.a_numpy(img)
    out = adaptive_wiener(arr, mysize=mysize, noise=noise, solo_cerebro=True)
    return io_utils.desde_numpy(out, ref=img)
