#!/usr/bin/env python3
"""
Módulo 7 v4 — ENSAMBLE de métodos (combina máscaras existentes, no crea métodos nuevos)
=======================================================================================
Combina las máscaras de la mejor variante v3 (componente ANCLADO a la semilla) de los
tres métodos mono-modales sembrados/umbralizados: Otsu(anclado), RegionGrowing, Watershed.
GMM multimodal queda FUERA del ensamble principal (sobre-segmenta ~1M vóxeles); se incluye
solo como sub-experimento opcional.

Estrategias de ensamble (global, la MISMA regla para todos los casos; el GT solo ubica la
semilla del post-proceso y evalúa):
  union        : ET si CUALQUIER método lo marca.
  voto_mayoria : ET si >= 2 de 3 coinciden.
  interseccion : ET si los 3 coinciden.
  union_post   : union + cierre (radio global) + componente conexo que CONTIENE la semilla.

Salidas (v4; preserva v1/v2/v3): metricas_ensamble_v4.csv, tabla_comparativa_v4.csv,
tabla_por_tamano_v4.csv, ranking_ensamble_v4.png, ensamble_v4_<caso>.png,
segmentacion_mejora_v4_reporte.html.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "04_segmentacion"))
sys.path.insert(0, str(ROOT / "07_segmentacion_mejora"))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import metricas as M             # noqa: E402
from comun import reporte as R              # noqa: E402
import seg_core as S                        # noqa: E402
from segmentacion_mejora import postproceso_anclado  # noqa: E402  (reusa v3)

import matplotlib.pyplot as plt             # noqa: E402

CIERRE_GLOBAL = 3   # radio de cierre global (el elegido por el barrido en v3)
METODOS_ENS = ["otsu", "regiongrowing", "watershed"]   # mono-modales (anclados, morf v3)
ESTRATEGIAS = ["union", "voto_mayoria", "interseccion", "union_post"]
# Referencia: mejor método individual de v3 (Dice medio sobre casos con ET).
REF_INDIVIDUAL = {"Otsu_anclado": 0.353, "RegionGrowing_anclado": 0.342}


def cargar_mascara(cid, metodo_archivo):
    p = C.OUTPUT_DIR / "segmentacion_mejora" / "outputs" / f"{cid}-{metodo_archivo}-morf-v3-ET.nii.gz"
    return io_zip.a_numpy(sitk.ReadImage(str(p))).astype(np.uint8)


def t1c_ref(cid):
    p = C.SALIDAS_MODULO["limpieza"] / "outputs" / f"{cid}-t1c_limpio.nii.gz"
    img = sitk.ReadImage(str(p), sitk.sitkFloat32)
    return img, io_zip.a_numpy(img)


def ensamblar(masks, ref, seed):
    """masks = [otsu, rg, ws] (uint8). Devuelve {estrategia: (mask, fallback)}."""
    stack = np.sum(masks, axis=0).astype(np.uint8)        # 0..3
    union = (stack >= 1).astype(np.uint8)
    voto = (stack >= 2).astype(np.uint8)
    inter = (stack == 3).astype(np.uint8)
    union_post, fb = postproceso_anclado(union, ref, CIERRE_GLOBAL, seed)
    return {"union": (union, False), "voto_mayoria": (voto, False),
            "interseccion": (inter, False), "union_post": (union_post, fb)}


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def _disp(sl):
    m = sl > 0
    if not m.any():
        return sl
    lo, hi = np.percentile(sl[m], [1, 99])
    out = np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1); out[~m] = 0
    return out


def figura_caso(cid, arr_t1c, seg, masks, union_post, figuras):
    from matplotlib.colors import ListedColormap
    et = seg == C.LABEL_ET
    z = int(round(ndi.center_of_mass(et)[0])) if et.any() else arr_t1c.shape[0] // 2
    cpred = ListedColormap([(0, 0, 0, 0), (1.0, 0.3, 0.3, 0.5)])
    paneles = [("Otsu anclado", masks[0]), ("RegionGrowing", masks[1]),
               ("Watershed", masks[2]), ("union_post", union_post)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (tt, mk) in zip(axes, paneles):
        ax.imshow(_disp(arr_t1c[z]), cmap="gray", origin="lower", aspect="auto")
        ax.imshow(mk[z], cmap=cpred, origin="lower", aspect="auto")
        if et.any():
            ax.contour(et[z].astype(float), levels=[0.5], colors=["#22D3EE"],
                       linewidths=0.8, origin="lower")
        ax.set_title(tt, fontsize=10); ax.axis("off")
    fig.suptitle(f"{cid} — ensamble (pred roja, GT cian)", fontsize=12); fig.tight_layout()
    return R.guardar_figura(fig, figuras / f"ensamble_v4_{cid}.png")


def figura_ranking(medias_ens, figuras):
    nombres = ESTRATEGIAS + list(REF_INDIVIDUAL.keys())
    vals = [medias_ens[e] for e in ESTRATEGIAS] + list(REF_INDIVIDUAL.values())
    colores = [C.PALETA["azul"]] * len(ESTRATEGIAS) + [C.PALETA["suave"]] * len(REF_INDIVIDUAL)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    barras = ax.bar(nombres, vals, color=colores)
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color=C.PALETA["rojo"], ls="--", lw=1, label="umbral 0,5")
    ax.set_ylabel("Dice medio (casos con ET)")
    ax.set_title("Ensamble vs mejor método individual (v3 anclado)")
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.legend(fontsize=8); fig.tight_layout()
    return R.guardar_figura(fig, figuras / "ranking_ensamble_v4.png")


def construir_reporte(tabla, tam, detalle, figs, fig_ranking, mejor, n_cruces,
                      supera, n_fallback, ruta_html):
    kpis = [
        R.kpi(mejor["estrategia"], "mejor estrategia"),
        R.kpi(f"{mejor['dice']:.3f}", "Dice medio"),
        R.kpi(f"{mejor['dice']-0.353:+.3f}", "vs mejor individual (0,353)"),
        R.kpi(str(n_cruces), "casos que cruzan 0,5"),
    ]
    secciones = []
    secciones.append(R.seccion(
        "Ensamble de métodos clásicos",
        "<p>Se combinan las máscaras de la mejor variante v3 (componente anclado) de "
        "<b>Otsu, RegionGrowing y Watershed</b> con tres reglas globales (unión, voto "
        "mayoría, intersección) más <b>union_post</b> (unión + cierre + componente que "
        "contiene la semilla). GMM queda fuera (sobre-segmenta). El GT solo ubica la "
        f"semilla del post-proceso y evalúa. Radio de cierre global = {CIERRE_GLOBAL}.</p>"
        f"<p>{supera}</p>"))
    secciones.append(R.seccion(
        "Dice medio por estrategia (+ mejor método individual de referencia)",
        R.df_a_tabla_html(tabla.round(4)),
        R.tarjeta_figura(R.png_a_base64(fig_ranking), "Estrategias vs individual; línea roja = 0,5.")))
    secciones.append(R.seccion("Dice medio por grupo de tamaño (mejor estrategia)",
                               R.df_a_tabla_html(tam.round(4))))
    bloques = [R.tarjeta_figura(R.png_a_base64(p), f"{cid}: 3 máscaras + union_post vs GT.")
               for cid, p in figs.items()]
    secciones.append(R.seccion("Ejemplos (caso bueno y malo)", *bloques))
    secciones.append(R.seccion("Detalle por caso y estrategia", R.df_a_tabla_html(detalle.round(4))))
    R.armar_reporte(
        "Ensamble de segmentación de ET (v4) — T1c · BraTS 2024 GLI",
        kpis, secciones,
        subtitulo="Módulo 7 v4 · unión / voto / intersección / unión+post-proceso anclado",
        ruta_salida=ruta_html)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    base_dir = C.OUTPUT_DIR / "segmentacion_mejora"
    outputs = base_dir / "outputs"; figuras = base_dir / "figuras"
    info = json.loads((C.SALIDAS_MODULO["segmentacion"] / "outputs" /
                       "casos_segmentacion.json").read_text(encoding="utf-8"))
    casos, casos_et = info["todos"], info["con_et"]
    print(f"[ENSAMBLE v4] {len(casos)} casos ({len(casos_et)} con ET)\n")

    detalle_filas = []
    masks_por_caso = {}
    n_fallback = 0
    for i, cid in enumerate(casos, 1):
        ref, arr_t1c = t1c_ref(cid)
        seg = io_zip.leer_seg_np(cid)
        et_presente = M.hay_et(seg)
        seed = S.semilla_et_brillante(arr_t1c, seg) or S.semilla_fallback(arr_t1c)
        masks = [cargar_mascara(cid, m) for m in METODOS_ENS]
        ens = ensamblar(masks, ref, seed)
        masks_por_caso[cid] = (arr_t1c, seg, masks, ens["union_post"][0])
        for estr, (mk, fb) in ens.items():
            if fb:
                n_fallback += 1
            io_zip.guardar_sitk(io_zip.desde_numpy(mk.astype(np.uint8), ref),
                                outputs / f"{cid}-ensamble-{estr}-v4-ET.nii.gz")
            ev = M.evaluar_et(mk, seg)
            detalle_filas.append({
                "case_id": cid, "estrategia": estr, "et_presente": et_presente,
                "dice": ev["dice"], "jaccard": ev["jaccard"],
                "sensibilidad": ev["sensibilidad"], "especificidad": ev["especificidad"],
                "pred_voxeles": int(mk.sum())})
        d_up = next(r["dice"] for r in detalle_filas
                    if r["case_id"] == cid and r["estrategia"] == "union_post")
        print(f"  [{i}/{len(casos)}] {cid}  union_post Dice="
              f"{'N/A' if d_up != d_up else round(d_up,3)}")
        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    detalle = pd.DataFrame(detalle_filas)
    detalle.to_csv(outputs / "metricas_ensamble_v4.csv", index=False)
    det_et = detalle[detalle.et_presente].copy()

    # Tabla comparativa: Dice medio por estrategia + referencia individual.
    medias_ens = det_et.groupby("estrategia")["dice"].mean().to_dict()
    filas_tabla = [{"fila": e, "tipo": "ensamble", "dice": round(medias_ens[e], 4)} for e in ESTRATEGIAS]
    for k, v in REF_INDIVIDUAL.items():
        filas_tabla.append({"fila": k, "tipo": "individual (v3)", "dice": v})
    tabla = pd.DataFrame(filas_tabla).sort_values("dice", ascending=False)
    tabla.to_csv(outputs / "tabla_comparativa_v4.csv", index=False)

    # Mejor estrategia de ensamble.
    mejor_estr = max(ESTRATEGIAS, key=lambda e: medias_ens[e])
    mejor = {"estrategia": mejor_estr, "dice": float(medias_ens[mejor_estr])}
    sens_mejor = float(det_et[det_et.estrategia == mejor_estr]["sensibilidad"].mean())
    espec_mejor = float(det_et[det_et.estrategia == mejor_estr]["especificidad"].mean())
    cruces = det_et[(det_et.estrategia == mejor_estr) & (det_et.dice >= 0.5)]
    n_cruces = int(len(cruces))
    mejor_ind = max(REF_INDIVIDUAL.values())
    if mejor["dice"] > mejor_ind:
        supera = (f"La mejor estrategia (<b>{mejor_estr}</b>, Dice {mejor['dice']:.3f}) "
                  f"SUPERA al mejor método individual ({mejor_ind:.3f}) por "
                  f"{mejor['dice']-mejor_ind:+.3f}.")
    else:
        supera = (f"Ninguna estrategia de ensamble supera al mejor método individual "
                  f"({mejor_ind:.3f}); la mejor es {mejor_estr} con {mejor['dice']:.3f} "
                  f"({mejor['dice']-mejor_ind:+.3f}).")

    # Por grupo de tamaño (mejor estrategia).
    eda = pd.read_csv(C.SALIDAS_MODULO["eda"] / "outputs" / "estadisticas_por_caso.csv")
    vol = {r.case_id: r.et_volumen_mm3 for r in eda.itertuples()}
    vols_et = np.array([vol[c] for c in casos_et], dtype=float)
    q1, q2 = np.percentile(vols_et, [33.33, 66.67])
    def grupo(cid):
        v = vol.get(cid, 0.0)
        return "pequeno" if v <= q1 else ("mediano" if v <= q2 else "grande")
    det_et["grupo"] = det_et["case_id"].map(grupo)
    tam = (det_et[det_et.estrategia == mejor_estr]
           .groupby("grupo")["dice"].mean().reindex(["pequeno", "mediano", "grande"]).reset_index())
    tam.columns = ["grupo", f"dice_{mejor_estr}"]
    tam.to_csv(outputs / "tabla_por_tamano_v4.csv", index=False)

    # Figuras: 1 caso bueno y 1 malo según union_post.
    up = det_et[det_et.estrategia == "union_post"].sort_values("dice")
    cid_malo, cid_bueno = up.iloc[0]["case_id"], up.iloc[-1]["case_id"]
    figs = {}
    for cid in [cid_bueno, cid_malo]:
        arr_t1c, seg, masks, union_post = masks_por_caso[cid]
        figs[cid] = figura_caso(cid, arr_t1c, seg, masks, union_post, figuras)
    fig_ranking = figura_ranking(medias_ens, figuras)

    construir_reporte(tabla, tam, detalle, figs, fig_ranking, mejor, n_cruces,
                      supera, n_fallback, base_dir / "segmentacion_mejora_v4_reporte.html")

    print("\n========== TABLA COMPARATIVA v4 (Dice medio) ==========")
    print(tabla.to_string(index=False))
    print(f"\n  {supera.replace('<b>','').replace('</b>','')}")
    print(f"  mejor estrategia: {mejor_estr}  Dice={mejor['dice']:.3f}  "
          f"sens={sens_mejor:.3f}  espec={espec_mejor:.3f}")
    print(f"  casos que cruzan 0,5 con {mejor_estr}: {n_cruces}")
    print(f"  fallback (semilla fuera de máscara): {n_fallback}")
    print(f"  salidas en: {base_dir}")
    print("=======================================================")


if __name__ == "__main__":
    main()
