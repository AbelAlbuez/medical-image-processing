#!/usr/bin/env python3
"""
Módulo 2 — LIMPIEZA de T1c (denoise -> N4 -> normalización percentil)
=====================================================================
Procesa SOLO la modalidad T1c de los casos demostrativos elegidos por el EDA
(lee output/eda/outputs/parametros_eda.json). Es la entrada única del módulo 4
(segmentación de ET).

Produce (en output/limpieza/):
  outputs/<caso>-t1c_limpio.nii.gz   -> volumen limpio (geometría preservada)
  figuras/limpieza_<caso>.png        -> antes/después (con panel ORIGINAL) por caso
  limpieza_reporte.html              -> reporte autocontenido

Ejecución:
  python limpieza.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import reporte as R              # noqa: E402
import limpieza_core as core                # noqa: E402

import matplotlib.pyplot as plt             # noqa: E402

# Parámetros de limpieza (documentados, no mágicos).
WIENER_SIZE = 3
N4_SHRINK = 4
PCTL_LO, PCTL_HI = 0.5, 99.5


def _corte_central_et(seg: np.ndarray, shape) -> int:
    """Índice z del centroide del ET (label 3); si no hay ET, corte central."""
    et = seg == C.LABEL_ET
    if et.any():
        return int(round(ndi.center_of_mass(et)[0]))
    return shape[0] // 2


def _disp(sl: np.ndarray) -> np.ndarray:
    """Reescala un corte a [0,1] por percentiles (1,99) dentro del cerebro, para mostrar."""
    m = sl > 0
    if not m.any():
        return sl
    lo, hi = np.percentile(sl[m], [1, 99])
    out = np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)
    out[~m] = 0
    return out


def figura_antes_despues(cid: str, arr_orig: np.ndarray, arr_limpio: np.ndarray,
                         seg: np.ndarray, figuras: Path) -> Path:
    """
    Panel por caso: T1c ORIGINAL | T1c LIMPIO (axial en el centroide del ET) +
    histograma de intensidades dentro del cerebro antes/después.
    """
    z = _corte_central_et(seg, arr_orig.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))

    axes[0].imshow(_disp(arr_orig[z]), cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("T1c original", fontsize=10); axes[0].axis("off")

    axes[1].imshow(arr_limpio[z], cmap="gray", origin="lower", aspect="auto")
    axes[1].set_title("T1c limpio (denoise→N4→norm)", fontsize=10); axes[1].axis("off")

    mo = arr_orig > 0
    ml = arr_limpio > 0
    ax = axes[2]
    if mo.any():
        vo = arr_orig[mo]
        ax.hist(vo / (np.percentile(vo, 99) or 1), bins=80, range=(0, 1.3),
                histtype="step", color=C.PALETA["suave"], label="original (p99)", density=True)
    if ml.any():
        ax.hist(arr_limpio[ml], bins=80, range=(0, 1.3), histtype="step",
                color=C.PALETA["azul"], label="limpio [0,1]", density=True)
    ax.set_title("Histograma cerebro", fontsize=10)
    ax.set_xlabel("Intensidad"); ax.legend(fontsize=8)

    fig.suptitle(f"{cid} — limpieza de T1c (corte z={z})", fontsize=12)
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / f"limpieza_{cid}.png")


def construir_reporte(tabla: pd.DataFrame, figs: dict, ruta_html: Path):
    kpis = [
        R.kpi(str(len(tabla)), "casos limpiados"),
        R.kpi("T1c", "modalidad"),
        R.kpi("Wiener→N4→%", "orden de limpieza"),
        R.kpi(f"[{PCTL_LO},{PCTL_HI}]", "percentiles → [0,1]"),
    ]
    secciones = []
    secciones.append(R.seccion(
        "Procedimiento",
        f"<p>Sobre cada caso demostrativo se procesa <b>solo T1c</b> con el orden "
        f"<b>denoise (Adaptive Wiener, ventana {WIENER_SIZE}) → N4 (shrink {N4_SHRINK}) → "
        f"normalización por percentiles [{PCTL_LO}, {PCTL_HI}] a [0,1]</b>. "
        f"La segmentación GT no se toca. El resultado <code>&lt;caso&gt;-t1c_limpio.nii.gz</code> "
        f"es la entrada del módulo de segmentación.</p>"))
    secciones.append(R.seccion(
        "Estadísticas por caso (dentro del cerebro)",
        R.df_a_tabla_html(tabla)))
    bloques_fig = [R.tarjeta_figura(R.png_a_base64(p), f"{cid}: original vs limpio + histograma.")
                   for cid, p in figs.items()]
    secciones.append(R.seccion("Antes / después por caso", *bloques_fig))
    R.armar_reporte(
        "Limpieza — T1c · BraTS 2024 GLI",
        kpis, secciones,
        subtitulo="Módulo 2 · Adaptive Wiener → N4 → normalización por percentiles",
        ruta_salida=ruta_html)


def main():
    dirs = C.dirs_modulo("limpieza")
    params_path = C.SALIDAS_MODULO["eda"] / "outputs" / "parametros_eda.json"
    if not params_path.exists():
        print(f"[ERROR] No existe {params_path}. Ejecuta primero el módulo 1 (EDA).")
        sys.exit(1)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    casos = params["casos_demostrativos"]
    print(f"[LIMPIEZA] casos demostrativos: {casos}\n")

    filas = []
    figs = {}
    for i, cid in enumerate(casos, 1):
        img = io_zip.leer_sitk(cid, "t1c")
        seg = io_zip.leer_seg_np(cid)
        if img is None or seg is None:
            print(f"  [{i}/{len(casos)}] {cid}  -> SIN T1c/seg, omitido")
            continue
        io_zip.verificar_geometria(img, cid)
        arr_orig = io_zip.a_numpy(img)

        print(f"  [{i}/{len(casos)}] {cid}  limpiando T1c (Wiener→N4→norm)...")
        img_limpio = core.limpiar_t1c(img, mysize_wiener=WIENER_SIZE,
                                      n4_shrink=N4_SHRINK, p_lo=PCTL_LO, p_hi=PCTL_HI)
        arr_limpio = io_zip.a_numpy(img_limpio)

        dst = dirs["outputs"] / f"{cid}-t1c_limpio.nii.gz"
        io_zip.guardar_sitk(img_limpio, dst)

        mo, ml = arr_orig > 0, arr_limpio > 0
        filas.append({
            "case_id": cid,
            "media_orig": round(float(arr_orig[mo].mean()), 1),
            "std_orig": round(float(arr_orig[mo].std()), 1),
            "p99_orig": round(float(np.percentile(arr_orig[mo], 99)), 1),
            "media_limpio": round(float(arr_limpio[ml].mean()), 4),
            "std_limpio": round(float(arr_limpio[ml].std()), 4),
            "min_limpio": round(float(arr_limpio[ml].min()), 4),
            "max_limpio": round(float(arr_limpio[ml].max()), 4),
            "archivo": dst.name,
        })
        figs[cid] = figura_antes_despues(cid, arr_orig, arr_limpio, seg, dirs["figuras"])
        print(f"        -> {dst.name}  (media_limpio={filas[-1]['media_limpio']}, "
              f"rango=[{filas[-1]['min_limpio']},{filas[-1]['max_limpio']}])")

        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    if not filas:
        print("\n[ERROR] No se limpió ningún caso.")
        sys.exit(1)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(dirs["outputs"] / "estadisticas_limpieza.csv", index=False)

    print("\n[LIMPIEZA] generando reporte HTML...")
    construir_reporte(tabla, figs, dirs["reporte"])

    print("\n========== RESUMEN LIMPIEZA ==========")
    print(f"  casos limpiados : {len(tabla)}")
    print(f"  volúmenes en    : {dirs['outputs']}")
    print(f"  figuras en      : {dirs['figuras']}")
    print(f"  reporte         : {dirs['reporte'].name}")
    print("======================================")


if __name__ == "__main__":
    main()
