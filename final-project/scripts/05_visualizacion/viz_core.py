"""
viz_core.py
===========
Overlays y mosaicos de 3 vistas (axial / sagital / coronal), `origin='lower'`,
centrados en el centroide del ET. Copiado/adaptado de viz_mosaics.py de Santiago,
con `aspect='auto'` uniforme y centrado en el ET (label 3).

Convención: arrays en orden numpy (z, y, x), tal como los devuelve
`sitk.GetArrayFromImage`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import constantes as C         # noqa: E402


def centroide(seg: np.ndarray) -> Tuple[int, int, int]:
    """Centroide (z,y,x) del ET (label 3); si no hay ET, del tumor (>0); si no, centro."""
    objetivo = (seg == C.LABEL_ET) if (seg == C.LABEL_ET).sum() > 0 else (seg > 0)
    if objetivo.sum() == 0:
        return tuple(s // 2 for s in seg.shape)
    return tuple(int(round(c)) for c in ndi.center_of_mass(objetivo))


def _norm8(sl: np.ndarray) -> np.ndarray:
    """Reescala un corte a [0,1] por percentiles (1,99) dentro del cerebro."""
    m = sl > 0
    if m.sum() == 0:
        return np.zeros_like(sl, dtype=np.float32)
    lo, hi = np.percentile(sl[m], [1, 99])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((sl - lo) / (hi - lo), 0, 1)
    out[~m] = 0
    return out


def _cortes_3vistas(vol: np.ndarray, c: Tuple[int, int, int]):
    """Devuelve (axial, coronal, sagital) en el centro c=(z,y,x)."""
    z, y, x = c
    return vol[z, :, :], vol[:, y, :], vol[:, :, x]


def overlay_3vistas(vol: np.ndarray, seg: np.ndarray,
                    pred: Optional[np.ndarray] = None,
                    c: Optional[Tuple[int, int, int]] = None,
                    titulo: str = "", path_out: Optional[Path] = None):
    """
    Figura con 3 paneles (axial/coronal/sagital): modalidad de fondo, GT (ET) como
    contorno cian y, si se da, la predicción como relleno rojo semitransparente.
    """
    c = c or centroide(seg)
    fondo = _cortes_3vistas(vol, c)
    gts = _cortes_3vistas((seg == C.LABEL_ET).astype(np.uint8), c)
    preds = _cortes_3vistas(pred, c) if pred is not None else (None, None, None)
    nombres = ["Axial", "Coronal", "Sagital"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    cmap_pred = ListedColormap([(0, 0, 0, 0), (1.0, 0.2, 0.2, 0.5)])
    for ax, bg, gt, pr, nm in zip(axes, fondo, gts, preds, nombres):
        ax.imshow(_norm8(bg), cmap="gray", origin="lower", aspect="auto")
        if gt.max() > 0:
            ax.contour(gt.astype(float), levels=[0.5], colors=["#22D3EE"],
                       linewidths=1.0, origin="lower")
        if pr is not None:
            ax.imshow(pr, cmap=cmap_pred, origin="lower", aspect="auto")
        ax.set_title(nm, fontsize=10); ax.axis("off")
    fig.suptitle(titulo, fontsize=12); fig.tight_layout()
    if path_out:
        Path(path_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_out, dpi=140, bbox_inches="tight"); plt.close(fig)
        return path_out
    return fig


def mosaico_grid_3vistas(vol: np.ndarray, seg: np.ndarray, preds: Dict[str, np.ndarray],
                         c: Optional[Tuple[int, int, int]] = None,
                         titulo: str = "", path_out: Optional[Path] = None):
    """
    Rejilla 3 (vistas: axial/coronal/sagital) × (1 + nº métodos): primera columna
    Original+GT (ET en cian), una columna por método con la predicción en rojo. Centrado
    en el centroide del ET. Es la figura comparativa principal por caso.
    """
    c = c or centroide(seg)
    vistas = ["Axial", "Coronal", "Sagital"]
    cols = ["Original + GT"] + list(preds.keys())
    et = (seg == C.LABEL_ET).astype(np.uint8)
    cmap_pred = ListedColormap([(0, 0, 0, 0), (1.0, 0.2, 0.2, 0.55)])
    fig, axes = plt.subplots(3, len(cols), figsize=(2.9 * len(cols), 8.4))
    for r in range(3):
        bg = _cortes_3vistas(vol, c)[r]
        gt = _cortes_3vistas(et, c)[r]
        ax = axes[r, 0]
        ax.imshow(_norm8(bg), cmap="gray", origin="lower", aspect="auto")
        if gt.max() > 0:
            ax.contour(gt.astype(float), levels=[0.5], colors=["#22D3EE"],
                       linewidths=1.0, origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(-0.10, 0.5, vistas[r], transform=ax.transAxes, rotation=90,
                va="center", ha="right", fontsize=11)
        if r == 0:
            ax.set_title(cols[0], fontsize=9)
        for ci, (nombre, pred) in enumerate(preds.items(), start=1):
            pr = _cortes_3vistas(pred, c)[r]
            a = axes[r, ci]
            a.imshow(_norm8(bg), cmap="gray", origin="lower", aspect="auto")
            a.imshow(pr, cmap=cmap_pred, origin="lower", aspect="auto")
            if gt.max() > 0:
                a.contour(gt.astype(float), levels=[0.5], colors=["#22D3EE"],
                          linewidths=0.6, origin="lower")
            a.axis("off")
            if r == 0:
                a.set_title(nombre, fontsize=9)
    fig.suptitle(titulo, fontsize=12); fig.tight_layout()
    if path_out:
        Path(path_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_out, dpi=140, bbox_inches="tight"); plt.close(fig)
        return path_out
    return fig


def mosaico_metodos(vol: np.ndarray, seg: np.ndarray, preds: Dict[str, np.ndarray],
                    vista: str = "axial", c: Optional[Tuple[int, int, int]] = None,
                    titulo: str = "", path_out: Optional[Path] = None):
    """
    Mosaico que compara los métodos en una vista: GT (cian) + cada predicción (roja).
    El primer panel es la imagen ORIGINAL con el GT; luego un panel por método.
    """
    c = c or centroide(seg)
    idx = {"axial": 0, "coronal": 1, "sagital": 2}[vista]
    bg = _cortes_3vistas(vol, c)[idx]
    gt = _cortes_3vistas((seg == C.LABEL_ET).astype(np.uint8), c)[idx]
    cmap_pred = ListedColormap([(0, 0, 0, 0), (1.0, 0.2, 0.2, 0.55)])
    n = len(preds) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.6))
    axes[0].imshow(_norm8(bg), cmap="gray", origin="lower", aspect="auto")
    if gt.max() > 0:
        axes[0].contour(gt.astype(float), levels=[0.5], colors=["#22D3EE"],
                        linewidths=1.0, origin="lower")
    axes[0].set_title("Original + GT (ET)", fontsize=9); axes[0].axis("off")
    for ax, (nombre, pred) in zip(axes[1:], preds.items()):
        pr = _cortes_3vistas(pred, c)[idx]
        ax.imshow(_norm8(bg), cmap="gray", origin="lower", aspect="auto")
        ax.imshow(pr, cmap=cmap_pred, origin="lower", aspect="auto")
        if gt.max() > 0:
            ax.contour(gt.astype(float), levels=[0.5], colors=["#22D3EE"],
                       linewidths=0.7, origin="lower")
        ax.set_title(nombre, fontsize=9); ax.axis("off")
    fig.suptitle(titulo, fontsize=12); fig.tight_layout()
    if path_out:
        Path(path_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_out, dpi=140, bbox_inches="tight"); plt.close(fig)
        return path_out
    return fig
