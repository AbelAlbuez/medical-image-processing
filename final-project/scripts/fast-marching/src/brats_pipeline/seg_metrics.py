"""
seg_metrics.py
==============
Métricas de solapamiento contra la segmentación de referencia (ground truth) y
construcción de la tabla comparativa de los métodos.

Dice    = 2|A n B| / (|A| + |B|)
Jaccard = |A n B| / |A u B|

Como los métodos clásicos producen una máscara binaria, comparamos contra las
sub-regiones anidadas de BraTS (config.SUBREGIONES_BRATS):
    WT = {1,2,3,4},  TC = {1,3,4},  ET = {3}
Así se ve qué método captura mejor cada sub-región (p.ej. Otsu-FLAIR ~ WT,
region-growing-T1c ~ ET).
"""
from __future__ import annotations
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from . import config


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0                      # ambos vacíos -> acuerdo perfecto
    return 2.0 * np.logical_and(a, b).sum() / s


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    u = np.logical_or(a, b).sum()
    if u == 0:
        return 1.0
    return np.logical_and(a, b).sum() / u


def mascara_subregion(seg: np.ndarray, etiquetas: Iterable[int]) -> np.ndarray:
    """Máscara binaria de una sub-región a partir del GT multi-etiqueta."""
    return np.isin(seg, list(etiquetas))


def evaluar(pred: np.ndarray, seg_gt: np.ndarray,
            subregiones: Dict = None) -> Dict[str, Dict[str, float]]:
    """
    Compara una predicción binaria contra cada sub-región de BraTS.

    Returns
    -------
    {sub: {"dice": d, "jaccard": j}}
    """
    subregiones = subregiones or config.SUBREGIONES_BRATS
    res = {}
    for sub, labs in subregiones.items():
        gt = mascara_subregion(seg_gt, labs)
        res[sub] = {"dice": dice(pred, gt), "jaccard": jaccard(pred, gt)}
    return res


def tabla_comparativa(resultados: Dict[str, Dict[str, np.ndarray]],
                      seg_gt: np.ndarray, case_id: str = "") -> pd.DataFrame:
    """
    Construye un DataFrame largo a partir de {nombre_metodo: pred_mask}.

    Columns: case_id, metodo, subregion, dice, jaccard
    """
    filas = []
    for metodo, pred in resultados.items():
        ev = evaluar(pred, seg_gt)
        for sub, m in ev.items():
            filas.append({"case_id": case_id, "metodo": metodo, "subregion": sub,
                          "dice": round(m["dice"], 4), "jaccard": round(m["jaccard"], 4)})
    return pd.DataFrame(filas)
