#!/usr/bin/env python3
"""
Módulo 7 v6 — ABLATION de la limpieza: ¿el preprocesamiento ayuda o daña la segmentación del ET?
================================================================================================
Aísla el efecto de Wiener / N4 / normalización comparando cuatro entradas de T1c:
  E1 CRUDO              : T1c del ZIP, sin limpieza.
  E2 SOLO N4            : solo N4 (shrink 4).
  E3 N4 + NORMALIZACIÓN : N4 + percentiles [0,5;99,5]->[0,1], SIN Wiener.
  E4 COMPLETO           : Wiener(3)->N4->norm (pipeline actual).

Dos estrategias (las mejores): Otsu anclado (v3) y ensamble union_post (v4), con post-proceso de
componente conectado anclado a la semilla. Umbrales RELATIVOS recalculados sobre cada entrada
(region growing por ventana [alpha*I_semilla, max] y Otsu por clase); la semilla brillante de ET se
recalcula sobre cada entrada. Sin umbral absoluto fijo entre versiones.

Reglas anti-trampa: GT solo semilla + evaluación; parámetros globales; sin DL.
Salidas con sufijo v6 (no pisan v1-v5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "02_limpieza"))
sys.path.insert(0, str(ROOT / "04_segmentacion"))
sys.path.insert(0, str(ROOT / "07_segmentacion_mejora"))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import metricas as M             # noqa: E402
from comun import reporte as R              # noqa: E402
import seg_core as S                        # noqa: E402
import limpieza_core as LC                  # noqa: E402
from segmentacion_mejora import postproceso_anclado  # noqa: E402

import matplotlib.pyplot as plt             # noqa: E402

CIERRE_GLOBAL = 3
RG_ALPHA = 0.65
ENTRADAS = ["E1_crudo", "E2_n4", "E3_n4norm", "E4_completo"]
ESTRATEGIAS = ["Otsu_anclado", "union_post"]


# --------------------------------------------------------------------------- #
# Construcción de las 4 entradas (todas derivadas del MISMO T1c crudo)
# --------------------------------------------------------------------------- #
def construir_entradas(cid):
    crudo = io_zip.leer_sitk(cid, "t1c")
    mask = io_zip.mascara_cerebro(crudo)
    n4 = LC.corregir_n4(crudo, mascara=mask, shrink_factor=4)
    n4norm = LC.normalizar_percentil_sitk(n4)
    completo = LC.limpiar_t1c(crudo)            # Wiener -> N4 -> norm
    return {"E1_crudo": crudo, "E2_n4": n4, "E3_n4norm": n4norm, "E4_completo": completo}


# --------------------------------------------------------------------------- #
# Region growing por ventana brillante con upper = max (robusto a la escala cruda)
# --------------------------------------------------------------------------- #
def rg_ventana(img: sitk.Image, arr: np.ndarray, seed, alpha=RG_ALPHA) -> np.ndarray:
    sx, sy, sz = int(seed[0]), int(seed[1]), int(seed[2])
    I = float(img.GetPixel((sx, sy, sz)))
    lo = alpha * I
    hi = float(arr.max()) + 1.0                 # ventana [alpha*I, max] (escala-robusto)
    rg = sitk.ConnectedThreshold(img, seedList=[(sx, sy, sz)], lower=float(lo), upper=hi)
    rg = sitk.BinaryMorphologicalClosing(rg, [1, 1, 1])
    return io_zip.a_numpy(sitk.Cast(rg, sitk.sitkUInt8)).astype(np.uint8)


def estrategias_sobre(img: sitk.Image, seg: np.ndarray):
    """Devuelve (otsu_anclado, union_post) sobre una entrada de T1c."""
    arr = io_zip.a_numpy(img)
    seed = S.semilla_et_brillante(arr, seg) or S.semilla_fallback(arr)
    otsu, _ = S.segmentar_otsu(arr, clases=3, tomar_clase="alta", nombre="")
    otsu_anc, _ = postproceso_anclado(otsu, img, CIERRE_GLOBAL, seed)
    rg = rg_ventana(img, arr, seed)
    rg_anc, _ = postproceso_anclado(rg, img, CIERRE_GLOBAL, seed)
    ws = S.segmentar_watershed(arr, seed=seed)
    ws_anc, _ = postproceso_anclado(ws, img, CIERRE_GLOBAL, seed)
    union = (np.sum([otsu_anc, rg_anc, ws_anc], axis=0) >= 1).astype(np.uint8)
    union_post, _ = postproceso_anclado(union, img, CIERRE_GLOBAL, seed)
    return otsu_anc, union_post


# --------------------------------------------------------------------------- #
# Separabilidad ET vs sano (FDR + Bhattacharyya) sobre una entrada
# --------------------------------------------------------------------------- #
def separabilidad(arr: np.ndarray, seg: np.ndarray):
    et = seg == C.LABEL_ET
    sano = (arr > 0) & (seg == 0)
    if et.sum() < 2 or sano.sum() < 2:
        return np.nan, np.nan
    ve, vs = arr[et], arr[sano]
    mu_e, var_e = float(ve.mean()), float(ve.var())
    mu_s, var_s = float(vs.mean()), float(vs.var())
    fdr = (mu_e - mu_s) ** 2 / max(var_e + var_s, 1e-8)
    v1, v2 = max(var_e, 1e-8), max(var_s, 1e-8)
    bhatt = 0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2)) + 0.25 * (mu_e - mu_s) ** 2 / (v1 + v2)
    return float(fdr), float(bhatt)


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def figura_dice(tabla, figuras):
    x = np.arange(len(ENTRADAS)); w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w/2, tabla["Otsu_anclado"], w, label="Otsu anclado", color=C.PALETA["azul"])
    ax.bar(x + w/2, tabla["union_post"], w, label="ensamble union_post", color=C.PALETA["verde"])
    ax.axhline(0.5, color=C.PALETA["rojo"], ls="--", lw=1, label="umbral 0,5")
    ax.set_xticks(x); ax.set_xticklabels(ENTRADAS, fontsize=9)
    ax.set_ylabel("Dice medio (casos con ET)")
    ax.set_title("Ablation de limpieza: Dice por entrada (umbral relativo)")
    ax.legend(fontsize=8); fig.tight_layout()
    return R.guardar_figura(fig, figuras / "ablation_dice_v6.png")


def figura_separabilidad(tabla, figuras):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(ENTRADAS, tabla["fdr_et"], color=C.PALETA["morado"])
    for i, v in enumerate(tabla["fdr_et"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("FDR ET vs sano (mediano)")
    ax.set_title("Separabilidad del ET por entrada de limpieza")
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / "ablation_separabilidad_v6.png")


def construir_reporte(tabla, detalle, fdice, fsep, respuestas, ruta_html):
    e1 = tabla.set_index("entrada").loc["E1_crudo", "union_post"]
    e4 = tabla.set_index("entrada").loc["E4_completo", "union_post"]
    kpis = [
        R.kpi(f"{e1:.3f}", "union_post E1 crudo"),
        R.kpi(f"{e4:.3f}", "union_post E4 completo"),
        R.kpi(f"{e4-e1:+.3f}", "E4 − E1 (efecto limpieza)"),
        R.kpi("estructural" if abs(e4 - e1) < 0.03 else "mixto", "naturaleza del techo"),
    ]
    secciones = [
        R.seccion("Diseño del ablation",
                  "<p>Cuatro entradas de T1c (crudo, solo N4, N4+norm sin Wiener, completo) con umbral "
                  "RELATIVO recalculado por entrada; dos estrategias (Otsu anclado, ensamble union_post). "
                  "El GT solo ubica la semilla y evalúa.</p>"),
        R.seccion("Dice medio por entrada (+ separabilidad del ET)",
                  R.df_a_tabla_html(tabla.round(4)),
                  R.tarjeta_figura(R.png_a_base64(fdice), "Dice por entrada y estrategia; línea 0,5."),
                  R.tarjeta_figura(R.png_a_base64(fsep), "FDR del ET por entrada.")),
        R.seccion("Respuestas", "<ul>" + "".join(f"<li>{r}</li>" for r in respuestas) + "</ul>"),
        R.seccion("Detalle por caso, entrada y estrategia", R.df_a_tabla_html(detalle.round(4))),
    ]
    R.armar_reporte("Ablation de limpieza para ET (v6) — T1c · BraTS 2024 GLI",
                    kpis, secciones,
                    subtitulo="Módulo 7 v6 · ¿Wiener/N4/normalización ayudan o dañan el Dice del ET?",
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
    print(f"[v6 ablation] {len(casos)} casos ({len(casos_et)} con ET)\n")

    filas, sep_filas = [], []
    for i, cid in enumerate(casos, 1):
        seg = io_zip.leer_seg_np(cid)
        et_presente = M.hay_et(seg)
        entradas = construir_entradas(cid)
        linea = []
        for ent, img in entradas.items():
            arr = io_zip.a_numpy(img)
            otsu_anc, union_post = estrategias_sobre(img, seg)
            for estr, mk in [("Otsu_anclado", otsu_anc), ("union_post", union_post)]:
                ev = M.evaluar_et(mk, seg)
                filas.append({"case_id": cid, "entrada": ent, "estrategia": estr,
                              "et_presente": et_presente, "dice": ev["dice"],
                              "jaccard": ev["jaccard"], "sensibilidad": ev["sensibilidad"],
                              "especificidad": ev["especificidad"], "pred_voxeles": int(mk.sum())})
            fdr, bhatt = separabilidad(arr, seg)
            sep_filas.append({"case_id": cid, "entrada": ent, "fdr": fdr, "bhatt": bhatt,
                              "et_presente": et_presente})
            up = next(f["dice"] for f in filas if f["case_id"] == cid and f["entrada"] == ent and f["estrategia"] == "union_post")
            linea.append(f"{ent}={'N/A' if up != up else round(up,3)}")
        print(f"  [{i}/{len(casos)}] {cid}  union_post: " + "  ".join(linea))
        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    detalle = pd.DataFrame(filas)
    detalle.to_csv(outputs / "metricas_ablation_v6.csv", index=False)
    sep = pd.DataFrame(sep_filas)

    det_et = detalle[detalle.et_presente]
    sep_et = sep[sep.et_presente]
    rows = []
    for ent in ENTRADAS:
        d = det_et[det_et.entrada == ent]
        s = sep_et[sep_et.entrada == ent]
        rows.append({
            "entrada": ent,
            "Otsu_anclado": round(float(d[d.estrategia == "Otsu_anclado"]["dice"].mean()), 4),
            "union_post": round(float(d[d.estrategia == "union_post"]["dice"].mean()), 4),
            "fdr_et": round(float(s["fdr"].median()), 4),
            "bhatt_et": round(float(s["bhatt"].median()), 4),
        })
    tabla = pd.DataFrame(rows)
    tabla.to_csv(outputs / "tabla_ablation_v6.csv", index=False)

    fdice = figura_dice(tabla, figuras)
    fsep = figura_separabilidad(tabla, figuras)

    # --- Respuestas a las 4 preguntas (con números reales) ---
    ti = tabla.set_index("entrada")
    up_e1, up_e4 = float(ti.loc["E1_crudo", "union_post"]), float(ti.loc["E4_completo", "union_post"])
    up_e3 = float(ti.loc["E3_n4norm", "union_post"])
    ot_e1, ot_e4 = float(ti.loc["E1_crudo", "Otsu_anclado"]), float(ti.loc["E4_completo", "Otsu_anclado"])
    fdr_e1, fdr_e4 = float(ti.loc["E1_crudo", "fdr_et"]), float(ti.loc["E4_completo", "fdr_et"])
    mejor_ent = tabla.loc[tabla["union_post"].idxmax(), "entrada"]
    dir_d = "sube" if up_e4 > up_e1 else ("baja" if up_e4 < up_e1 else "no cambia")
    r1 = (f"(1) El Dice de union_post {dir_d} de E1 (crudo {up_e1:.3f}) a E4 (completo {up_e4:.3f}); "
          f"magnitud {up_e4-up_e1:+.3f}. (Otsu: {ot_e1:.3f}->{ot_e4:.3f}).")
    r2 = (f"(2) E3 sin Wiener = {up_e3:.3f} vs E4 con Wiener = {up_e4:.3f} -> "
          f"{'Wiener atenúa el ET' if up_e3 > up_e4 else 'Wiener no daña' if up_e3 <= up_e4 else ''}. "
          f"E1 crudo {up_e1:.3f} vs E4 {up_e4:.3f} -> "
          f"{'la normalización daña' if up_e1 > up_e4 else 'la normalización no daña'}.")
    r3 = (f"(3) FDR del ET: crudo {fdr_e1:.2f} vs completo {fdr_e4:.2f} "
          f"({'baja' if fdr_e4 < fdr_e1 else 'sube' if fdr_e4 > fdr_e1 else 'igual'} con la limpieza).")
    r4 = (f"(4) Mejor entrada para union_post: {mejor_ent}. La diferencia E4-E1 es "
          f"{up_e4-up_e1:+.3f}: el techo es "
          f"{'estructural (la limpieza casi no mueve el Dice)' if abs(up_e4-up_e1) < 0.03 else 'parcialmente atribuible a la limpieza'}.")
    respuestas = [r1, r2, r3, r4]

    construir_reporte(tabla, detalle, fdice, fsep, respuestas,
                      base_dir / "segmentacion_mejora_v6_ablation_reporte.html")

    print("\n========== TABLA ABLATION v6 (Dice medio + separabilidad por entrada) ==========")
    print(tabla.to_string(index=False))
    print()
    for r in respuestas:
        print("  " + r)
    print(f"  salidas en: {base_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()
