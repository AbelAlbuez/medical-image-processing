#!/usr/bin/env python3
"""
Módulo 5 — VISUALIZACIÓN y análisis post-mortem
===============================================
Cierra el pipeline: overlays y mosaicos de 3 vistas (axial/sagital/coronal, origin='lower')
centrados en el centroide del ET, con el GT y las 4 segmentaciones lado a lado, más un
análisis post-mortem de por qué cada método se comporta como se observa.

USA EL MISMO SET DE 9 CASOS de output/segmentacion/outputs/casos_segmentacion.json y lee
las métricas ya calculadas (no recalcula). Este módulo es de ANÁLISIS: NO genera .nii.gz.

Produce (en output/visualizacion/):
  figuras/viz_grid_<caso>.png      -> 3 vistas × (Original+GT + 4 métodos)
  figuras/viz_overlay_<caso>.png   -> overlay 3 vistas del mejor método (RegionGrowing)
  figuras/viz_ranking.png          -> ranking de métodos por Dice
  visualizacion_reporte.html       -> reporte integrador (estilo comun/reporte.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import reporte as R              # noqa: E402
import viz_core as V                        # noqa: E402

import matplotlib.pyplot as plt             # noqa: E402

METODOS = ["Otsu", "RegionGrowing", "Watershed", "GMM_multimodal"]
NOMBRE_ARCHIVO = {"Otsu": "otsu", "RegionGrowing": "regiongrowing",
                  "Watershed": "watershed", "GMM_multimodal": "gmm_multimodal"}
MEJOR_METODO = "RegionGrowing"


def cargar_pred(cid, metodo, seg_out):
    p = seg_out / f"{cid}-{NOMBRE_ARCHIVO[metodo]}-ET.nii.gz"
    return io_zip.a_numpy(sitk.ReadImage(str(p))).astype(np.uint8)


def figura_ranking(resumen, figuras):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colores = [C.PALETA["morado"] if mm else C.PALETA["azul"] for mm in resumen["multimodal"]]
    barras = ax.bar(resumen["metodo"], resumen["dice"], color=colores)
    for b, v in zip(barras, resumen["dice"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Dice medio (casos con ET)")
    ax.set_title("Comparación final de métodos — Dice (ET)")
    ax.set_ylim(0, max(0.05, float(resumen["dice"].max()) * 1.25))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C.PALETA["azul"], label="mono-modal (T1c)"),
                       Patch(color=C.PALETA["morado"], label="multimodal")], fontsize=8)
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / "viz_ranking.png")


def analisis_postmortem(resumen):
    r = resumen.set_index("metodo")
    def g(m, col): return float(r.loc[m, col])
    return (
        "<ul>"
        f"<li><b>RegionGrowing gana</b> (Dice {g('RegionGrowing','dice'):.3f}): al sembrar en el "
        "vóxel de ET más brillante y crecer solo por tejido brillante <code>[α·I, max]</code>, "
        "captura el realce de forma localizada y con la mayor especificidad útil "
        f"({g('RegionGrowing','especificidad'):.3f}) sin fugarse al resto del cerebro.</li>"
        f"<li><b>GMM multimodal sobre-segmenta</b> (sensibilidad {g('GMM_multimodal','sensibilidad'):.3f} "
        f"pero Dice {g('GMM_multimodal','dice'):.3f}): al agrupar T1c+T2w+T2f marca como 'tumor' un "
        "clúster amplio que incluye edema y vasos; recupera mucho ET (alta sensibilidad) a costa de "
        f"muchos falsos positivos (especificidad {g('GMM_multimodal','especificidad'):.3f}, la más baja).</li>"
        f"<li><b>Otsu falla por unimodalidad</b> (Dice {g('Otsu','dice'):.3f}): el histograma de T1c es "
        "unimodal (coef. bimodalidad &lt; 5/9 en los 9 casos, avisado por el módulo 4), así que el "
        "umbral de la clase alta cae en un punto arbitrario y mezcla realce con tejido sano brillante.</li>"
        f"<li><b>Watershed</b> (Dice {g('Watershed','dice'):.3f}) es intermedio: depende del marcador; "
        "con la semilla en el ET brillante delimita cuencas razonables, pero es sensible al ruido "
        "residual y a la heterogeneidad del realce.</li>"
        "<li><b>Hallazgo de la semilla:</b> en ET post-tratamiento el realce forma un anillo alrededor "
        "de la cavidad de resección; el <i>centroide</i> del ET cae en esa cavidad (tejido oscuro, "
        "label 4) y arruinaba el crecimiento. Por eso se siembra en el <b>vóxel de ET más brillante</b> "
        "(inicialización semi-automática desde el GT); la máscara final sale del crecimiento sobre T1c, "
        "no del GT.</li>"
        "</ul>")


def construir_reporte(resumen, figs_caso, fig_ranking, casos, sin_et, n_et, ruta_html):
    mejor = resumen.iloc[0]
    kpis = [
        R.kpi(str(mejor["metodo"]), "mejor método (Dice)"),
        R.kpi(f"{mejor['dice']:.3f}", "Dice del mejor método"),
        R.kpi(str(n_et), "casos con ET"),
        R.kpi(str(len(casos)), "casos visualizados"),
    ]
    secciones = []
    secciones.append(R.seccion(
        "Comparación final de los 4 métodos",
        R.df_a_tabla_html(resumen.round(4)),
        R.tarjeta_figura(R.png_a_base64(fig_ranking),
                         "Dice medio por método (morado = multimodal).")))
    secciones.append(R.seccion("Análisis post-mortem", analisis_postmortem(resumen)))
    # Galería por caso
    bloques = []
    for cid in casos:
        etiqueta = " (SIN ET — ilustra falsos positivos)" if cid in sin_et else ""
        bloques.append(f"<h3 style='margin:14px 0 4px'>{cid}{etiqueta}</h3>")
        bloques.append(R.tarjeta_figura(R.png_a_base64(figs_caso[cid]["grid"]),
                       "3 vistas × (Original+GT y los 4 métodos)."))
        bloques.append(R.tarjeta_figura(R.png_a_base64(figs_caso[cid]["overlay"]),
                       f"Overlay 3 vistas — {MEJOR_METODO} (predicción roja, GT cian)."))
    secciones.append(R.seccion("Mosaicos por caso (3 vistas, centrados en el ET)", *bloques))
    secciones.append(R.seccion(
        "Caso sin ET — limitación",
        f"<p>El caso <b>{sin_et[0]}</b> no tiene ET en el GT. Aun así, los 4 métodos producen "
        "máscaras (falsos positivos): ningún método clásico 'sabe' abstenerse cuando no hay realce. "
        "Es la limitación esperada y motiva un paso de decisión 'hay/no hay ET' previo en un sistema real.</p>"))
    R.armar_reporte(
        "Visualización y análisis — ET · BraTS 2024 GLI",
        kpis, secciones,
        subtitulo="Módulo 5 · Mosaicos 3 vistas + análisis post-mortem de los 4 métodos",
        ruta_salida=ruta_html)


def main():
    dirs = C.dirs_modulo("visualizacion")
    seg_out = C.SALIDAS_MODULO["segmentacion"] / "outputs"
    limp_out = C.SALIDAS_MODULO["limpieza"] / "outputs"

    casos_json = seg_out / "casos_segmentacion.json"
    resumen_csv = seg_out / "metricas_resumen.csv"
    if not casos_json.exists() or not resumen_csv.exists():
        print("[ERROR] Faltan salidas del módulo 4 (casos_segmentacion.json / metricas_resumen.csv).")
        sys.exit(1)
    info = json.loads(casos_json.read_text(encoding="utf-8"))
    casos, sin_et = info["todos"], info["sin_et"]
    resumen = pd.read_csv(resumen_csv)
    print(f"[VIZ] {len(casos)} casos (mismos del módulo 4)\n")

    figs_caso = {}
    for i, cid in enumerate(casos, 1):
        t1c_path = limp_out / f"{cid}-t1c_limpio.nii.gz"
        if not t1c_path.exists():
            print(f"  [{i}/{len(casos)}] {cid}  -> SIN T1c limpio, omitido")
            continue
        vol = io_zip.a_numpy(sitk.ReadImage(str(t1c_path), sitk.sitkFloat32))
        seg = io_zip.leer_seg_np(cid)
        preds = {m: cargar_pred(cid, m, seg_out) for m in METODOS}
        c = V.centroide(seg)

        grid = V.mosaico_grid_3vistas(vol, seg, preds, c=c,
                                      titulo=f"{cid} — comparación de métodos (centrado en ET)",
                                      path_out=dirs["figuras"] / f"viz_grid_{cid}.png")
        overlay = V.overlay_3vistas(vol, seg, pred=preds[MEJOR_METODO], c=c,
                                    titulo=f"{cid} — {MEJOR_METODO} (pred roja) + GT (cian)",
                                    path_out=dirs["figuras"] / f"viz_overlay_{cid}.png")
        figs_caso[cid] = {"grid": grid, "overlay": overlay}
        print(f"  [{i}/{len(casos)}] {cid}  -> viz_grid + viz_overlay")
        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    if not figs_caso:
        print("\n[ERROR] No se generó ninguna figura.")
        sys.exit(1)

    fig_ranking = figura_ranking(resumen, dirs["figuras"])
    n_et = len(info["con_et"])
    casos_ok = [c for c in casos if c in figs_caso]
    print("\n[VIZ] generando reporte HTML...")
    construir_reporte(resumen, figs_caso, fig_ranking, casos_ok, sin_et, n_et, dirs["reporte"])

    print("\n========== RESUMEN VISUALIZACIÓN ==========")
    print(f"  casos visualizados : {len(casos_ok)}")
    print(f"  figuras grid       : {len(figs_caso)}  (+ overlays + ranking)")
    print(f"  reporte            : {dirs['reporte'].name}")
    print(f"  salidas en         : {dirs['base']}")
    print("===========================================")


if __name__ == "__main__":
    main()
