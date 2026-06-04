"""
metricas.py
===========
Métricas de solapamiento contra la segmentación de referencia (ground truth) y
construcción de tablas comparativas entre métodos.

Base copiada de seg_metrics.py de Santiago (dice, jaccard, mascara_subregion,
tabla_comparativa) y EXTENDIDA con **sensibilidad** y **especificidad**, que el
proyecto necesita y faltaban.

Definiciones (A = predicción binaria, B = GT binario; TP/FP/FN/TN sobre los vóxeles):
    Dice          = 2|A∩B| / (|A| + |B|)
    Jaccard       = |A∩B| / |A∪B|
    Sensibilidad  = TP / (TP + FN)        (recall; fracción de ET realmente detectado)
    Especificidad = TN / (TN + FP)        (fracción de tejido sano correctamente excluido)

Nuestro foco es la sub-región ET = {3}. `evaluar` reporta por las tres sub-regiones
anidadas para contexto, pero el módulo de segmentación se centra en ET.

Manejo de ET vacío (caso sin seg==3): cuando el GT está vacío, Dice/Jaccard de un
método que también predice vacío valen 1.0 (acuerdo perfecto); sensibilidad y Dice
NO están bien definidas si se quiere medir detección -> se devuelven como NaN para
marcarlos como N/A en las tablas. La especificidad sí está definida.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from . import constantes as C


# --------------------------------------------------------------------------- #
# Métricas escalares
# --------------------------------------------------------------------------- #
def _confusion(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, int]:
    """Devuelve (TP, FP, FN, TN) entre predicción `a` y GT `b` (booleanos)."""
    a = a.astype(bool)
    b = b.astype(bool)
    tp = int(np.logical_and(a, b).sum())
    fp = int(np.logical_and(a, ~b).sum())
    fn = int(np.logical_and(~a, b).sum())
    tn = int(np.logical_and(~a, ~b).sum())
    return tp, fp, fn, tn


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0                      # ambos vacíos -> acuerdo perfecto
    return float(2.0 * np.logical_and(a, b).sum() / s)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    u = np.logical_or(a, b).sum()
    if u == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / u)


def sensibilidad(a: np.ndarray, b: np.ndarray) -> float:
    """TP / (TP + FN). NaN si el GT está vacío (no hay positivos que detectar)."""
    tp, _fp, fn, _tn = _confusion(a, b)
    denom = tp + fn
    if denom == 0:
        return float("nan")
    return tp / denom


def especificidad(a: np.ndarray, b: np.ndarray) -> float:
    """TN / (TN + FP). NaN si no hay negativos en el GT (caso degenerado)."""
    _tp, fp, _fn, tn = _confusion(a, b)
    denom = tn + fp
    if denom == 0:
        return float("nan")
    return tn / denom


# --------------------------------------------------------------------------- #
# Sub-regiones BraTS
# --------------------------------------------------------------------------- #
def mascara_subregion(seg: np.ndarray, etiquetas: Iterable[int]) -> np.ndarray:
    """Máscara binaria de una sub-región a partir del GT multi-etiqueta."""
    return np.isin(seg, list(etiquetas))


def mascara_et(seg: np.ndarray) -> np.ndarray:
    """Atajo: máscara binaria del Tumor Realzante (ET = label 3)."""
    return seg == C.LABEL_ET


def hay_et(seg: np.ndarray) -> bool:
    """True si el caso tiene algún vóxel de ET (seg==3)."""
    return bool((seg == C.LABEL_ET).any())


# --------------------------------------------------------------------------- #
# Evaluación de una predicción
# --------------------------------------------------------------------------- #
def evaluar_et(pred: np.ndarray, seg_gt: np.ndarray) -> Dict[str, float]:
    """
    Evalúa una predicción binaria contra el ET del GT. Devuelve
    {'dice','jaccard','sensibilidad','especificidad','et_presente'}.

    Si el GT no tiene ET, Dice/Jaccard/sensibilidad se devuelven como NaN (N/A) para
    no contaminar promedios; se conserva 'especificidad' y la bandera 'et_presente'.
    """
    gt = mascara_et(seg_gt)
    presente = bool(gt.any())
    if not presente:
        return {
            "dice": float("nan"),
            "jaccard": float("nan"),
            "sensibilidad": float("nan"),
            "especificidad": especificidad(pred, gt),
            "et_presente": False,
        }
    return {
        "dice": dice(pred, gt),
        "jaccard": jaccard(pred, gt),
        "sensibilidad": sensibilidad(pred, gt),
        "especificidad": especificidad(pred, gt),
        "et_presente": True,
    }


def evaluar_subregiones(pred: np.ndarray, seg_gt: np.ndarray,
                        subregiones: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
    """
    Compara una predicción binaria contra cada sub-región de BraTS (WT/TC/ET) con las
    cuatro métricas. Útil para contexto en el EDA/visualización.
    """
    subregiones = subregiones or C.SUBREGIONES_BRATS
    res: Dict[str, Dict[str, float]] = {}
    for sub, labs in subregiones.items():
        gt = mascara_subregion(seg_gt, labs)
        res[sub] = {
            "dice": dice(pred, gt),
            "jaccard": jaccard(pred, gt),
            "sensibilidad": sensibilidad(pred, gt),
            "especificidad": especificidad(pred, gt),
        }
    return res


# --------------------------------------------------------------------------- #
# Tablas comparativas
# --------------------------------------------------------------------------- #
def tabla_metodos_et(predicciones: Dict[str, np.ndarray], seg_gt: np.ndarray,
                     case_id: str = "", multimodales: Iterable[str] = ()) -> pd.DataFrame:
    """
    Construye un DataFrame por método para un caso, evaluando contra ET.

    Parameters
    ----------
    predicciones : {nombre_metodo: máscara_binaria}
    seg_gt : GT multi-etiqueta del caso.
    case_id : id del caso (columna).
    multimodales : nombres de métodos que usan varias modalidades (se marcan en la tabla).

    Columns: case_id, metodo, multimodal, et_presente, dice, jaccard, sensibilidad, especificidad
    """
    multimodales = set(multimodales)
    filas = []
    for metodo, pred in predicciones.items():
        ev = evaluar_et(pred, seg_gt)
        filas.append({
            "case_id": case_id,
            "metodo": metodo,
            "multimodal": metodo in multimodales,
            "et_presente": ev["et_presente"],
            "dice": _redondear(ev["dice"]),
            "jaccard": _redondear(ev["jaccard"]),
            "sensibilidad": _redondear(ev["sensibilidad"]),
            "especificidad": _redondear(ev["especificidad"]),
        })
    return pd.DataFrame(filas)


def resumen_por_metodo(tabla: pd.DataFrame) -> pd.DataFrame:
    """
    Promedia las métricas por método sobre los casos CON ET presente (ignora NaN).
    Conserva la bandera `multimodal`. Ordena por Dice descendente.
    """
    cols = ["dice", "jaccard", "sensibilidad", "especificidad"]
    agg = (tabla.groupby("metodo")
                .agg({**{c: "mean" for c in cols}, "multimodal": "first"})
                .round(4)
                .reset_index()
                .sort_values("dice", ascending=False, na_position="last"))
    return agg


def _redondear(x: float, n: int = 4) -> float:
    """Redondea conservando NaN (para marcar N/A en las tablas)."""
    return float("nan") if (x is None or np.isnan(x)) else round(float(x), n)
