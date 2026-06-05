"""
normalize.py
============
Normalización de intensidades.

Por qué hace falta (Sección D del reporte EDA): la intensidad de RM no tiene
unidades físicas fijas, y entre casos del GLI la media por modalidad varía con un
**coeficiente de variación de ~51-60%** (T1c 59.9%, T2w 56.2%, T2-FLAIR 59.2%,
T1n 51.4%). Sin normalizar, "intensidad 800" significa cosas distintas en cada
caso y cualquier umbral o modelo entrenado no transfiere. Normalizar pone a todos
"en lo mismo".

Esquemas disponibles:
  * "zscore"     : (x - media_cerebro) / std_cerebro   [recomendado, estándar BraTS/nnU-Net]
  * "percentil"  : recorta a [p_lo, p_hi] dentro del cerebro y reescala a [0, 1]

En ambos casos solo se usan los vóxeles de cerebro (mask = x > 0) para estimar los
estadísticos, y el fondo se mantiene en 0.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np

import SimpleITK as sitk
from . import io_utils


def zscore_en_mascara(volumen: np.ndarray) -> np.ndarray:
    """z-score usando solo vóxeles de cerebro (x>0). Fondo queda en 0."""
    v = np.asarray(volumen, dtype=np.float32)
    m = v > 0
    if m.sum() == 0:
        return v
    mu = float(v[m].mean())
    sd = float(v[m].std())
    sd = sd if sd > 1e-8 else 1.0
    out = np.zeros_like(v)
    out[m] = (v[m] - mu) / sd
    return out


def percentil_a_unidad(volumen: np.ndarray,
                       p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
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


def normalizar_imagen(img: sitk.Image, esquema: str = "zscore",
                      **kwargs) -> sitk.Image:
    """Wrapper SITK->SITK. `esquema` in {'zscore','percentil'}."""
    arr = io_utils.a_numpy(img)
    if esquema == "zscore":
        out = zscore_en_mascara(arr)
    elif esquema == "percentil":
        out = percentil_a_unidad(arr, **kwargs)
    else:
        raise ValueError(f"Esquema desconocido: {esquema!r} (usa 'zscore' o 'percentil').")
    return io_utils.desde_numpy(out.astype(np.float32), ref=img)
