"""
viz_mosaics.py
==============
Overlays y mosaicos con 3 vistas (axial / sagital / coronal), `origin='lower'`,
centrados en el tumor.

Convenciones:
  * Trabajamos sobre arrays en orden numpy (z, y, x) tal como los devuelve
    `sitk.GetArrayFromImage` / `nibabel.get_fdata().T`. Ajusta `cargar()` a tu
    fuente. Aquí asumimos arrays (z, y, x).
  * El centro se toma del centroide del GT (preferentemente ET=3, si no, todo el
    tumor) o de una semilla dada.
  * `origin='lower'` en todos los `imshow`, como pide el enunciado.
"""
from __future__ import annotations
import os
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")           # backend sin pantalla (Colab/servidor); quítalo si usas %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def centroide(seg: np.ndarray) -> Tuple[int, int, int]:
    """Centroide (z,y,x) del ET (label 3); si no hay, de todo el tumor (>0)."""
    objetivo = (seg == 3) if (seg == 3).sum() > 0 else (seg > 0)
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
    axial   = vol[z, :, :]
    coronal = vol[:, y, :]
    sagital = vol[:, :, x]
    return axial, coronal, sagital


def overlay_3vistas(vol: np.ndarray, seg: np.ndarray,
                    pred: Optional[np.ndarray] = None,
                    c: Optional[Tuple[int, int, int]] = None,
                    titulo: str = "", path_out: Optional[str] = None):
    """
    Una figura con 3 paneles (axial/coronal/sagital). Muestra la modalidad de fondo,
    el GT como contorno y, si se da, la predicción como relleno semitransparente.
    """
    c = c or centroide(seg)
    fondo = _cortes_3vistas(vol, c)
    gts   = _cortes_3vistas(seg, c)
    preds = _cortes_3vistas(pred, c) if pred is not None else (None, None, None)
    nombres = ["Axial", "Coronal", "Sagital"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    cmap_pred = ListedColormap([(0, 0, 0, 0), (1.0, 0.3, 0.3, 0.45)])  # transparente / rojo
    for ax, bg, gt, pr, nm in zip(axes, fondo, gts, preds, nombres):
        ax.imshow(_norm8(bg), cmap="gray", origin="lower")
        # GT como contornos por etiqueta
        for lab in np.unique(gt):
            if lab == 0:
                continue
            ax.contour((gt == lab).astype(float), levels=[0.5], colors=["#22D3EE"],
                       linewidths=0.8, origin="lower")
        if pr is not None:
            ax.imshow(pr, cmap=cmap_pred, origin="lower")
        ax.set_title(nm, fontsize=10)
        ax.axis("off")
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    if path_out:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        fig.savefig(path_out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path_out
    return fig


def mosaico_metodos(vol: np.ndarray, seg: np.ndarray,
                    preds: Dict[str, np.ndarray],
                    vista: str = "axial",
                    c: Optional[Tuple[int, int, int]] = None,
                    titulo: str = "", path_out: Optional[str] = None):
    """
    Mosaico que compara varios métodos en una misma vista/corte: GT + cada predicción.
    `preds` = {nombre_metodo: mask}.
    """
    c = c or centroide(seg)
    idx = {"axial": 0, "coronal": 1, "sagital": 2}[vista]
    bg = _cortes_3vistas(vol, c)[idx]
    gt = _cortes_3vistas(seg, c)[idx]
    cmap_pred = ListedColormap([(0, 0, 0, 0), (1.0, 0.3, 0.3, 0.5)])

    n = len(preds) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    axes[0].imshow(_norm8(bg), cmap="gray", origin="lower")
    for lab in np.unique(gt):
        if lab:
            axes[0].contour((gt == lab).astype(float), levels=[0.5], colors=["#22D3EE"],
                            linewidths=0.8, origin="lower")
    axes[0].set_title("Ground truth"); axes[0].axis("off")
    for ax, (nombre, pred) in zip(axes[1:], preds.items()):
        pr = _cortes_3vistas(pred, c)[idx]
        ax.imshow(_norm8(bg), cmap="gray", origin="lower")
        ax.imshow(pr, cmap=cmap_pred, origin="lower")
        ax.contour((gt > 0).astype(float), levels=[0.5], colors=["#22D3EE"],
                   linewidths=0.6, origin="lower")
        ax.set_title(nombre, fontsize=9); ax.axis("off")
    fig.suptitle(titulo, fontsize=12); fig.tight_layout()
    if path_out:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        fig.savefig(path_out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path_out
    return fig


# ------------------------------------------------------------------ #
# Visualizaciones específicas para ET por sustracción                #
# ------------------------------------------------------------------ #

def figura_mapa_diferencia(arr_t1n: np.ndarray,
                            arr_t1c: np.ndarray,
                            mapa_dif: np.ndarray,
                            gt_et: np.ndarray,
                            pred: np.ndarray,
                            umbral: float,
                            case_id: str = "",
                            path_out: Optional[str] = None):
    """
    Figura de 4 paneles que explica el método de sustracción:
      1. T1n (sin contraste)
      2. T1c (con contraste)
      3. Mapa T1c − T1n (señal del gadolinio)
      4. Overlay: T1c + GT-ET (cian) + predicción sustracción (rojo)

    Centrada en el corte axial con más GT-ET (o central si GT vacío).
    """
    sumas_z = gt_et.sum(axis=(1, 2))
    z = int(np.argmax(sumas_z)) if sumas_z.max() > 0 else arr_t1c.shape[0] // 2

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(_norm8(arr_t1n[z]), cmap="gray", origin="lower")
    axes[0].set_title("T1n  (sin contraste)", fontsize=10)

    axes[1].imshow(_norm8(arr_t1c[z]), cmap="gray", origin="lower")
    axes[1].set_title("T1c  (con contraste)", fontsize=10)

    vmax = np.percentile(mapa_dif[mapa_dif > 0], 99) if (mapa_dif > 0).any() else 1
    im = axes[2].imshow(np.clip(mapa_dif[z], 0, None),
                        cmap="hot", vmin=0, vmax=vmax, origin="lower")
    axes[2].set_title(f"T1c − T1n  (umbral={umbral:.3f})", fontsize=10)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(_norm8(arr_t1c[z]), cmap="gray", origin="lower")
    if gt_et[z].any():
        axes[3].contour(gt_et[z].astype(float), levels=[0.5],
                        colors=["#22d3ee"], linewidths=1.5, origin="lower")
    if pred[z].any():
        ov = np.zeros((*pred[z].shape, 4))
        ov[pred[z] > 0] = [0.94, 0.27, 0.27, 0.5]
        axes[3].imshow(ov, origin="lower")
    axes[3].set_title("Overlay: cian=GT · rojo=pred", fontsize=10)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{case_id}  —  z={z}", fontsize=12)
    fig.tight_layout()

    if path_out:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        fig.savefig(path_out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path_out
    return fig


def figura_comparativa_metodos(arr_t1c: np.ndarray,
                                gt_et: np.ndarray,
                                mascaras: Dict[str, np.ndarray],
                                case_id: str = "",
                                path_out: Optional[str] = None):
    """
    Mosaico flexible con fondo T1c.
    Soporta cualquier numero de metodos (2 paneles fijos + N metodos).
    mascaras debe incluir "_mapa_dif" como clave interna.
    """
    COLORES = {
        "sustraccion":    (0.94, 0.27, 0.27),   # rojo
        "otsu_T1c":       (0.98, 0.57, 0.09),   # naranja
        "gmm_T1c":        (0.66, 0.33, 0.97),   # morado
        "gmm_2d":         (0.95, 0.77, 0.06),   # amarillo
        "rango_doble":    (0.13, 0.94, 0.53),   # verde brillante
        "fast_marching":  (0.06, 0.73, 0.97),   # cyan - metodo principal
        "semilla":        (0.23, 0.51, 0.98),   # azul
        "region_growing": (0.23, 0.51, 0.98),   # azul
        "interseccion":   (0.13, 0.77, 0.37),   # verde
        # Contornos deformables (spline / level set)
        "level_set":          (0.06, 0.73, 0.97),   # cyan
        "variational_spline": (0.96, 0.45, 0.71),   # rosa
        "bspline":            (0.55, 0.86, 0.20),   # lima
        "spline":             (1.00, 0.84, 0.00),   # dorado
    }

    mapa_dif = mascaras.get("_mapa_dif", np.zeros_like(arr_t1c))
    sumas_z  = gt_et.sum(axis=(1, 2))
    z = int(np.argmax(sumas_z)) if sumas_z.max() > 0 else arr_t1c.shape[0] // 2

    # Calcular grid segun numero de metodos (excluir claves internas "_*")
    metodos_plot = [k for k in COLORES.keys() if k in mascaras]
    n_metodos = len(metodos_plot)
    n_total = 2 + n_metodos  # 2 paneles fijos + metodos
    ncols = min(4, n_total)
    nrows = (n_total + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4.5))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Panel 0: T1c + GT contorno
    axes[0].imshow(_norm8(arr_t1c[z]), cmap="gray", origin="lower")
    if gt_et[z].any():
        axes[0].contour(gt_et[z].astype(float), levels=[0.5],
                        colors=["#22d3ee"], linewidths=1.8, origin="lower")
    axes[0].set_title("T1c  +  GT-ET (cian)", fontsize=10)

    # Panel 1: mapa diferencia
    vmax = np.percentile(mapa_dif[mapa_dif > 0], 99) if (mapa_dif > 0).any() else 1
    axes[1].imshow(_norm8(arr_t1c[z]), cmap="gray", alpha=0.45, origin="lower")
    im = axes[1].imshow(np.clip(mapa_dif[z], 0, None),
                        cmap="hot", alpha=0.85, vmin=0, vmax=vmax, origin="lower")
    axes[1].set_title("Mapa T1c − T1n", fontsize=10)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Paneles 2+: métodos (solo los que existen en mascaras)
    for ax, nombre in zip(axes[2:], metodos_plot):
        rgb = COLORES[nombre]
        pred = mascaras.get(nombre)
        ax.imshow(_norm8(arr_t1c[z]), cmap="gray", origin="lower")
        if gt_et[z].any():
            ax.contour(gt_et[z].astype(float), levels=[0.5],
                       colors=["#22d3ee"], linewidths=1.2, origin="lower")
        if pred is not None and pred[z].any():
            ov = np.zeros((*pred[z].shape, 4))
            ov[pred[z] > 0] = [*rgb, 0.52]
            ax.imshow(ov, origin="lower")
        ax.set_title(nombre, fontsize=9)

    for ax in axes:
        ax.axis("off")
    # Ocultar paneles sobrantes si el grid tiene mas celdas que metodos
    for ax in axes[n_total:]:
        ax.set_visible(False)
    fig.suptitle(f"{case_id}  —  axial z={z}  (cian=GT | color=pred)", fontsize=11)
    fig.tight_layout()

    if path_out:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        fig.savefig(path_out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path_out
    return fig


def figura_metricas_resumen(df,
                             path_out: Optional[str] = None):
    """
    Barras de Dice-ET por método y caso a partir del DataFrame de métricas.
    df debe tener columnas: case_id, metodo, dice_ET
    """
    import pandas as pd
    metodos  = df["metodo"].unique()
    casos    = df["case_id"].unique()
    x        = np.arange(len(casos))
    ancho    = 0.18
    COLORES_BAR = {
        "sustraccion":    "#ef4444",
        "otsu_T1c":       "#f97316",
        "gmm_T1c":        "#a855f7",
        "region_growing": "#3b82f6",
    }

    fig, ax = plt.subplots(figsize=(max(8, len(casos) * 2), 5))
    for i, metodo in enumerate(metodos):
        vals = [df[(df["case_id"] == c) & (df["metodo"] == metodo)]["dice_ET"].values
                for c in casos]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        color = COLORES_BAR.get(metodo, "#6b7280")
        bars = ax.bar(x + i * ancho, vals, ancho, label=metodo,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + ancho * (len(metodos) - 1) / 2)
    ax.set_xticklabels([c.split("-")[-2] + "-" + c.split("-")[-1]
                        for c in casos], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Dice-ET", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Dice-ET por caso y método", fontsize=12)
    fig.tight_layout()

    if path_out:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        fig.savefig(path_out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path_out
    return fig
