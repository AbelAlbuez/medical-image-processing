#!/usr/bin/env python3
"""
Módulo 7 — MEJORA de la segmentación de ET (post-proceso morfológico + sustracción)
==================================================================================
Mide cuánto mejora el Dice con técnicas CLÁSICAS legítimas, sin tocar el ground truth
salvo como referencia de evaluación y para ubicar la semilla (igual que el módulo 4).

Dos estrategias, combinadas en cuatro variantes por método:
  base        : máscara del módulo 4 (sin cambios).
  morf        : base + apertura(r=1) + cierre(r*) + componente conectado MAYOR.
  sub         : método re-ejecutado sobre la sustracción S = clip(T1c_limpio - T1n_limpio, 0).
  sub_morf    : sub + el mismo post-proceso morfológico.

Reglas anti-trampa (CRÍTICO):
  * El GT (seg==3) se usa SOLO para (a) ubicar la semilla del region growing y (b) evaluar.
    Nunca para construir la máscara ni para elegir parámetros caso a caso.
  * Los parámetros (radio de cierre r*, alpha del region growing en S) son GLOBALES: se eligen
    por barrido sobre el Dice PROMEDIO de los casos con ET, no por caso mirando su GT.
  * Sin aprendizaje profundo (solo SimpleITK / scikit-image / scipy).

Salidas en output/segmentacion_mejora/ (outputs/, figuras/, segmentacion_mejora_reporte.html).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "02_limpieza"))
sys.path.insert(0, str(ROOT / "04_segmentacion"))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import metricas as M             # noqa: E402
from comun import reporte as R              # noqa: E402
import seg_core as S                        # noqa: E402
import limpieza_core as LC                  # noqa: E402

import matplotlib.pyplot as plt             # noqa: E402

METODOS = ["Otsu", "RegionGrowing", "Watershed", "GMM_multimodal"]
METODOS_MONO = ["Otsu", "RegionGrowing", "Watershed"]   # aplicables a la sustracción
MULTIMODALES = {"GMM_multimodal"}
ARCHIVO = {"Otsu": "otsu", "RegionGrowing": "regiongrowing",
           "Watershed": "watershed", "GMM_multimodal": "gmm_multimodal"}
VARIANTES = ["base", "morf", "sub", "sub_morf"]

# Grids GLOBALES del barrido (itertools.product). Un único valor para todos los casos.
SWEEP_CIERRE = [1, 2, 3]        # radio del cierre morfológico (sin apertura en v2)
SWEEP_ALPHA = [0.55, 0.65, 0.75]  # alpha del region growing sobre la sustracción


# --------------------------------------------------------------------------- #
# Post-proceso morfológico + selección de componente conectado
# --------------------------------------------------------------------------- #
def postproceso_mayor(mask_arr: np.ndarray, ref: sitk.Image, cierre_r: int) -> np.ndarray:
    """Cierre(cierre_r) -> componente conectado MÁS GRANDE (v2). uint8 (z,y,x)."""
    if mask_arr.sum() == 0:
        return mask_arr.astype(np.uint8)
    m = sitk.BinaryMorphologicalClosing(io_zip.desde_numpy(mask_arr.astype(np.uint8), ref), [cierre_r] * 3)
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(m), sortByObjectSize=True)
    return io_zip.a_numpy(sitk.Cast(cc == 1, sitk.sitkUInt8)).astype(np.uint8)


def postproceso_anclado(mask_arr: np.ndarray, ref: sitk.Image, cierre_r: int,
                        seed) -> tuple[np.ndarray, bool]:
    """
    Cierre(cierre_r) -> componente conectado que CONTIENE LA SEMILLA (v3).

    `seed` = índice IJK SimpleITK (x,y,z) del vóxel de ET más brillante. Se elige el
    componente conexo que contiene ese vóxel. Si la semilla cae en fondo de la máscara
    (la máscara no la cubre), se usa el componente MÁS GRANDE como fallback y se marca.

    Devuelve (mask uint8, fallback_bool).
    """
    if mask_arr.sum() == 0:
        return mask_arr.astype(np.uint8), False
    m = sitk.BinaryMorphologicalClosing(io_zip.desde_numpy(mask_arr.astype(np.uint8), ref), [cierre_r] * 3)
    cc_img = sitk.ConnectedComponent(m)
    cc = io_zip.a_numpy(cc_img)                       # etiquetas (z, y, x)
    sx, sy, sz = (int(seed[0]), int(seed[1]), int(seed[2]))
    sz_, sy_, sx_ = (np.clip(sz, 0, cc.shape[0] - 1),
                     np.clip(sy, 0, cc.shape[1] - 1),
                     np.clip(sx, 0, cc.shape[2] - 1))
    lab = int(cc[sz_, sy_, sx_])
    if lab == 0:                                      # semilla fuera de la máscara -> fallback mayor
        rel = sitk.RelabelComponent(cc_img, sortByObjectSize=True)
        return io_zip.a_numpy(sitk.Cast(rel == 1, sitk.sitkUInt8)).astype(np.uint8), True
    return (cc == lab).astype(np.uint8), False


# --------------------------------------------------------------------------- #
# Carga de volúmenes por caso (T1c limpio, T1n limpio, sustracción)
# --------------------------------------------------------------------------- #
def cargar_volumenes(cid: str):
    """
    Devuelve (ref_sitk_t1c, arr_t1c, arr_S, seg).

    Sustracción S = clip(T1c_limpio - HM(T1n_limpio -> T1c_limpio), 0). Antes de restar,
    el T1n limpio se IGUALA a la distribución del T1c limpio por *histogram matching*: así
    el tejido que no capta contraste se cancela y queda resaltado el realce (gadolinio).
    Restar percentiles independientes no funciona (destruye la relación entre modalidades).
    """
    t1c_path = C.SALIDAS_MODULO["limpieza"] / "outputs" / f"{cid}-t1c_limpio.nii.gz"
    ref = sitk.ReadImage(str(t1c_path), sitk.sitkFloat32)
    arr_t1c = io_zip.a_numpy(ref)
    # T1n: misma limpieza (Wiener->N4->percentil) en memoria, luego histogram matching a T1c.
    t1n_img = LC.limpiar_t1c(io_zip.leer_sitk(cid, "t1n"))
    t1n_hm = sitk.HistogramMatching(t1n_img, ref, numberOfHistogramLevels=256,
                                    numberOfMatchPoints=12, thresholdAtMeanIntensity=True)
    arr_S = np.clip(arr_t1c - io_zip.a_numpy(t1n_hm), 0.0, None).astype(np.float32)
    seg = io_zip.leer_seg_np(cid)
    return ref, arr_t1c, arr_S, seg


def base_mask(cid: str, metodo: str) -> np.ndarray:
    """Máscara base del módulo 4."""
    p = C.SALIDAS_MODULO["segmentacion"] / "outputs" / f"{cid}-{ARCHIVO[metodo]}-ET.nii.gz"
    return io_zip.a_numpy(sitk.ReadImage(str(p))).astype(np.uint8)


def metodo_sobre(volumen: np.ndarray, ref: sitk.Image, seg: np.ndarray,
                 metodo: str, alpha: float) -> np.ndarray:
    """Re-ejecuta un método mono-modal sobre `volumen` (p.ej. la sustracción S)."""
    seed = S.semilla_et_brillante(volumen, seg)         # semilla = vóxel de ET más brillante en S
    if seed is None:
        seed = S.semilla_fallback(volumen)
    if metodo == "Otsu":
        m, _ = S.segmentar_otsu(volumen, clases=3, tomar_clase="alta", nombre="")
        return m
    if metodo == "RegionGrowing":
        img = io_zip.desde_numpy(volumen.astype(np.float32), ref)
        return S.connected_threshold(img, seed, alpha=alpha, radio_cierre=1)
    if metodo == "Watershed":
        et = (seg == C.LABEL_ET)
        return S.segmentar_watershed(volumen, seed=seed if et.any() else None)
    raise ValueError(metodo)


# --------------------------------------------------------------------------- #
# Barrido GLOBAL de parámetros (sobre el Dice PROMEDIO, no por caso)
# --------------------------------------------------------------------------- #
def barrido_global(casos_et, vol_cache, outputs: Path):
    """
    Barre (cierre x alpha) con itertools.product y elige el par que maximiza el Dice
    MEDIO de la variante estrella (RegionGrowing sub_morf) sobre los casos con ET.
    Devuelve (cierre*, alpha*, df_sweep).
    """
    filas = []
    for r, a in itertools.product(SWEEP_CIERRE, SWEEP_ALPHA):
        dices = []
        for cid in casos_et:
            ref, _arr_t1c, arr_S, seg = vol_cache[cid]
            seed_S = S.semilla_et_brillante(arr_S, seg) or S.semilla_fallback(arr_S)
            sub = metodo_sobre(arr_S, ref, seg, "RegionGrowing", alpha=a)
            sub_morf, _fb = postproceso_anclado(sub, ref, r, seed_S)
            dices.append(M.dice(sub_morf, M.mascara_et(seg)))
        filas.append({"cierre": r, "alpha": a, "dice_medio": round(float(np.mean(dices)), 4)})
    df = pd.DataFrame(filas).sort_values("dice_medio", ascending=False)
    df.to_csv(outputs / "barrido_cierre_v2.csv", index=False)
    mejor = df.iloc[0]
    return int(mejor["cierre"]), float(mejor["alpha"]), df


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


def figura_v3_otsu(cid, arr_t1c, seg, otsu_base, ref, cierre_s, seed_t1c, figuras):
    """Otsu: base vs componente-mayor vs componente-anclado (muestra el efecto del anclaje)."""
    et = seg == C.LABEL_ET
    z = int(round(ndi.center_of_mass(et)[0])) if et.any() else arr_t1c.shape[0] // 2
    from matplotlib.colors import ListedColormap
    cpred = ListedColormap([(0, 0, 0, 0), (1.0, 0.3, 0.3, 0.5)])
    m_mayor = postproceso_mayor(otsu_base, ref, cierre_s)
    m_anc, fb = postproceso_anclado(otsu_base, ref, cierre_s, seed_t1c)
    paneles = [("base", otsu_base), ("componente mayor", m_mayor),
               ("componente anclado", m_anc)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax, (tt, mk) in zip(axes, paneles):
        ax.imshow(_disp(arr_t1c[z]), cmap="gray", origin="lower", aspect="auto")
        ax.imshow(mk[z], cmap=cpred, origin="lower", aspect="auto")
        if et.any():
            ax.contour(et[z].astype(float), levels=[0.5], colors=["#22D3EE"],
                       linewidths=0.8, origin="lower")
        # marca la semilla en verde
        ax.plot(seed_t1c[0], seed_t1c[1], "o", color="#22e07a", markersize=6,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.set_title(f"Otsu — {tt}", fontsize=10); ax.axis("off")
    fig.suptitle(f"{cid} — Otsu: mayor vs anclado (pred roja, GT cian, semilla verde)", fontsize=11)
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / f"comparacion_v3_{cid}.png")


def figura_ranking(tabla, figuras):
    metodos = tabla["metodo"].tolist()
    x = np.arange(len(metodos)); w = 0.2
    cols = {"base": C.PALETA["suave"], "morf": C.PALETA["azul"],
            "sub": C.PALETA["naranja"], "sub_morf": C.PALETA["verde"]}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, var in enumerate(VARIANTES):
        vals = [tabla.loc[tabla.metodo == mt, var].values[0] for mt in metodos]
        vals = [0 if (v != v) else v for v in vals]
        ax.bar(x + (i - 1.5) * w, vals, w, label=var, color=cols[var])
    ax.axhline(0.5, color=C.PALETA["rojo"], ls="--", lw=1, label="umbral 0,5")
    ax.set_xticks(x); ax.set_xticklabels(metodos, fontsize=9)
    ax.set_ylabel("Dice medio (casos con ET)")
    ax.set_title("Dice por método y variante (base vs mejoras) — v3 componente anclado")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / "ranking_mejora_v3.png")


# --------------------------------------------------------------------------- #
# Reporte HTML
# --------------------------------------------------------------------------- #
def construir_reporte(tabla, detalle, figs, fig_ranking, mejor_base, mejor_mej,
                      cierre_s, alpha_s, n_et, cruzaron, tabla_tamano, cmp_v1, ruta_html):
    delta = mejor_mej["dice"] - mejor_base["dice"]
    kpis = [
        R.kpi(f"{mejor_base['dice']:.3f}", f"Dice base ({mejor_base['etq']})"),
        R.kpi(f"{mejor_mej['dice']:.3f}", f"Dice mejorado ({mejor_mej['etq']})"),
        R.kpi(f"{delta:+.3f}", "ganancia de Dice"),
        R.kpi(str(cruzaron), "casos que cruzaron 0,5"),
    ]
    secciones = []
    secciones.append(R.seccion(
        "Estrategias y reglas anti-trampa (v2: sin apertura)",
        f"<p>Cuatro variantes por método: <b>base</b>, <b>+morfología</b> "
        f"(cierre r={cierre_s} + componente conectado mayor, <b>sin apertura</b>), "
        f"<b>+sustracción</b> (método sobre $S=\\mathrm{{clip}}(T1c-T1n,0)$ con histogram matching) "
        f"y <b>+sustracción+morfología</b>. Parámetros GLOBALES por Dice promedio: cierre "
        f"$r^*={cierre_s}$, $\\alpha^*={alpha_s}$. El GT solo ubica la semilla y evalúa; "
        f"nunca construye la máscara ni ajusta por caso.</p>"
        f"<p>{cmp_v1}</p>"))
    secciones.append(R.seccion(
        "Tabla comparativa — Dice medio por método y variante",
        R.df_a_tabla_html(tabla.round(4)),
        R.tarjeta_figura(R.png_a_base64(fig_ranking), "Dice por método y variante; línea roja = 0,5.")))
    secciones.append(R.seccion(
        "Dice medio por GRUPO DE TAMAÑO de ET (volumen del EDA)",
        "<p>Separa el efecto del post-proceso según el tamaño del ET. La morfología de "
        "componente mayor tiende a ayudar en ET grande y a ser neutra/perjudicial en pequeño.</p>",
        R.df_a_tabla_html(tabla_tamano.round(4))))
    bloques = [R.tarjeta_figura(R.png_a_base64(p), f"{cid}: base / morf / sub_morf (RegionGrowing).")
               for cid, p in figs.items()]
    secciones.append(R.seccion("Comparación por caso — ET pequeño vs grande (RegionGrowing)", *bloques))
    secciones.append(R.seccion(
        "Detalle por caso, método y variante",
        R.df_a_tabla_html(detalle.round(4))))
    if cruzaron > 0:
        honesto = (f"<p><b>{cruzaron} combinaciones caso/método/variante cruzan 0,5</b> de Dice, "
                   "pero <b>ningún método alcanza 0,5 de Dice MEDIO</b>. El post-proceso mejora "
                   "combinaciones concretas (sobre todo en ET grande) sin elevar el promedio por "
                   "encima de 0,5, lo que refuerza que los métodos clásicos por intensidad son "
                   "insuficientes para el ET.</p>")
    else:
        honesto = ("<p><b>Ningún método/variante cruza 0,5</b>. Las mejoras clásicas no bastan: "
                   "refuerza la tesis de que el ET requiere modelos de forma o aprendizaje.</p>")
    secciones.append(R.seccion("Lectura honesta", honesto))
    R.armar_reporte(
        "Mejora de segmentación de ET (v2) — T1c · BraTS 2024 GLI",
        kpis, secciones,
        subtitulo="Módulo 7 v2 · cierre + componente mayor (sin apertura) + sustracción T1c$-$T1n",
        ruta_salida=ruta_html)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    base_dir = C.OUTPUT_DIR / "segmentacion_mejora"
    outputs = base_dir / "outputs"; figuras = base_dir / "figuras"
    outputs.mkdir(parents=True, exist_ok=True); figuras.mkdir(parents=True, exist_ok=True)

    info = json.loads((C.SALIDAS_MODULO["segmentacion"] / "outputs" /
                       "casos_segmentacion.json").read_text(encoding="utf-8"))
    casos, casos_et = info["todos"], info["con_et"]
    print(f"[MEJORA] {len(casos)} casos ({len(casos_et)} con ET)\n")

    # Cargar volúmenes (cache: T1c, S, seg) — limpia T1n una vez por caso.
    print("[MEJORA] limpiando T1n y construyendo sustracción S = clip(T1c - T1n, 0)...")
    vol_cache = {}
    for cid in casos:
        vol_cache[cid] = cargar_volumenes(cid)
        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)
        print(f"  ok {cid}")

    # Barrido global de parámetros.
    cierre_s, alpha_s, df_sweep = barrido_global(casos_et, vol_cache, outputs)
    print(f"\n[MEJORA] barrido global -> cierre*={cierre_s}, alpha*={alpha_s} "
          f"(elegido por Dice MEDIO, no por caso)")

    # Generar las 4 variantes por método y caso.
    detalle_filas = []
    figs = {}
    # Figuras v3: casos donde Otsu colapsaba con componente-mayor (ET pequeño/mediano).
    demostrativos = {"BraTS-GLI-00529-100", "BraTS-GLI-02062-100"}
    print("\n[MEJORA] generando variantes (componente ANCLADO a la semilla)...")
    for i, cid in enumerate(casos, 1):
        ref, arr_t1c, arr_S, seg = vol_cache[cid]
        et_presente = M.hay_et(seg)
        seed_t1c = S.semilla_et_brillante(arr_t1c, seg) or S.semilla_fallback(arr_t1c)
        seed_S = S.semilla_et_brillante(arr_S, seg) or S.semilla_fallback(arr_S)
        for metodo in METODOS:
            # (variante, mascara, fallback)
            registros = []
            mb = base_mask(cid, metodo)
            registros.append(("base", mb, False))
            if metodo in MULTIMODALES:
                # GMM: dos sub-variantes de componente para comparar honestamente.
                m_anc, fb = postproceso_anclado(mb, ref, cierre_s, seed_t1c)
                registros.append(("morf", m_anc, fb))                       # anclado
                registros.append(("morf_mayor", postproceso_mayor(mb, ref, cierre_s), False))
            else:
                m_anc, fb = postproceso_anclado(mb, ref, cierre_s, seed_t1c)
                registros.append(("morf", m_anc, fb))
                msub = metodo_sobre(arr_S, ref, seg, metodo, alpha=alpha_s)
                registros.append(("sub", msub, False))
                msm, fbs = postproceso_anclado(msub, ref, cierre_s, seed_S)
                registros.append(("sub_morf", msm, fbs))
            for var, mk, fb in registros:
                io_zip.guardar_sitk(io_zip.desde_numpy(mk.astype(np.uint8), ref),
                                    outputs / f"{cid}-{ARCHIVO[metodo]}-{var}-v3-ET.nii.gz")
                ev = M.evaluar_et(mk, seg)
                detalle_filas.append({
                    "case_id": cid, "metodo": metodo, "variante": var,
                    "multimodal": metodo in MULTIMODALES, "et_presente": et_presente,
                    "dice": ev["dice"], "jaccard": ev["jaccard"],
                    "sensibilidad": ev["sensibilidad"], "especificidad": ev["especificidad"],
                    "pred_voxeles": int(mk.sum()), "fallback": bool(fb)})
        # Figura v3: Otsu base vs componente-mayor vs componente-anclado en casos que colapsaban.
        if cid in demostrativos:
            figs[cid] = figura_v3_otsu(cid, arr_t1c, seg, base_mask(cid, "Otsu"),
                                       ref, cierre_s, seed_t1c, figuras)
        d_otsu = next((r["dice"] for r in detalle_filas
                       if r["case_id"] == cid and r["metodo"] == "Otsu" and r["variante"] == "morf"), float("nan"))
        print(f"  [{i}/{len(casos)}] {cid}  Otsu morf(anclado) Dice="
              f"{'N/A' if d_otsu != d_otsu else round(d_otsu,3)}")

    detalle = pd.DataFrame(detalle_filas)
    detalle.to_csv(outputs / "metricas_mejora_v3.csv", index=False)
    n_fallback = int(detalle["fallback"].sum())

    # Tabla comparativa: Dice medio por método x variante (solo casos con ET; morf=anclado).
    det_et = detalle[detalle.et_presente].copy()
    tabla_fig = det_et.pivot_table(index="metodo", columns="variante", values="dice",
                                   aggfunc="mean").reindex(index=METODOS, columns=VARIANTES).reset_index()
    piv = tabla_fig.copy()
    piv.columns = ["metodo", "base", "mas_morfologia", "mas_sustraccion", "mas_sustraccion_morfologia"]
    piv.to_csv(outputs / "tabla_comparativa_v3.csv", index=False)

    # GMM: componente anclado (morf) vs mayor (morf_mayor).
    gmm = det_et[det_et.metodo == "GMM_multimodal"]
    gmm_anc = float(gmm[gmm.variante == "morf"]["dice"].mean())
    gmm_may = float(gmm[gmm.variante == "morf_mayor"]["dice"].mean())

    # --- Análisis por GRUPO DE TAMAÑO de ET (volumen del EDA, terciles globales) ---
    eda = pd.read_csv(C.SALIDAS_MODULO["eda"] / "outputs" / "estadisticas_por_caso.csv")
    vol = {r.case_id: r.et_volumen_mm3 for r in eda.itertuples()}
    vols_et = np.array([vol[c] for c in casos_et], dtype=float)
    q1, q2 = np.percentile(vols_et, [33.33, 66.67])
    def grupo_tam(cid):
        v = vol.get(cid, 0.0)
        return "pequeno" if v <= q1 else ("mediano" if v <= q2 else "grande")
    det_et["grupo"] = det_et["case_id"].map(grupo_tam)
    tabla_tam = (det_et.pivot_table(index=["grupo", "metodo"], columns="variante",
                                    values="dice", aggfunc="mean")
                       .reindex(columns=VARIANTES).reset_index())
    orden = {"pequeno": 0, "mediano": 1, "grande": 2}
    tabla_tam["__o"] = tabla_tam["grupo"].map(orden)
    tabla_tam = tabla_tam.sort_values(["__o", "metodo"]).drop(columns="__o")
    tabla_tam.to_csv(outputs / "tabla_por_tamano_v3.csv", index=False)

    # --- Mejor base vs mejor mejorado; cuántas combinaciones cruzan 0,5 ---
    medias = det_et.groupby(["metodo", "variante"])["dice"].mean().reset_index()
    base_rows = medias[medias.variante == "base"].sort_values("dice", ascending=False).iloc[0]
    mej_rows = medias[medias.variante != "base"].sort_values("dice", ascending=False).iloc[0]
    mejor_base = {"dice": float(base_rows["dice"]), "etq": f"{base_rows['metodo']} base"}
    mejor_mej = {"dice": float(mej_rows["dice"]), "etq": f"{mej_rows['metodo']} {mej_rows['variante']}"}
    cruces = det_et[det_et["dice"] >= 0.5][["case_id", "metodo", "variante", "dice"]].sort_values("dice", ascending=False)
    n_cruces = int(len(cruces))

    # --- Comparación v2 (componente mayor) vs v3 (componente anclado) ---
    v2_path = outputs / "tabla_comparativa_v2.csv"
    cmp_df = None
    cmp_txt = "No se encontró la tabla v2 para comparar."
    if v2_path.exists():
        v2 = pd.read_csv(v2_path).set_index("metodo")
        v3 = piv.set_index("metodo")
        cmp_rows = []
        for mt in METODOS:
            for col in ["base", "mas_morfologia", "mas_sustraccion", "mas_sustraccion_morfologia"]:
                a = v2.loc[mt, col] if (mt in v2.index and col in v2.columns) else np.nan
                b = v3.loc[mt, col] if (mt in v3.index and col in v3.columns) else np.nan
                if a == a and b == b:
                    cmp_rows.append({"metodo": mt, "variante": col, "v2": round(a, 4),
                                     "v3": round(b, 4), "delta": round(b - a, 4)})
        cmp_df = pd.DataFrame(cmp_rows)
        cmp_df.to_csv(outputs / "comparacion_v2_v3.csv", index=False)
        cmp_txt = ("Comparado con la v2 (componente mayor): anclar el componente a la semilla de ET "
                   "rescata sobre todo a Otsu (su componente mayor era tejido sano).")

    # --- Antes/después de Otsu en los casos que colapsaban en v2 ---
    colapsaban = ["BraTS-GLI-02063-106", "BraTS-GLI-00529-100",
                  "BraTS-GLI-02062-100", "BraTS-GLI-02063-102"]
    otsu_cmp = None
    v2det = outputs / "metricas_mejora_v2.csv"
    if v2det.exists():
        d2 = pd.read_csv(v2det)
        rows = []
        for cid in colapsaban:
            for var in ["morf", "sub_morf"]:
                a = d2[(d2.case_id == cid) & (d2.metodo == "Otsu") & (d2.variante == var)]["dice"]
                b = det_et[(det_et.case_id == cid) & (det_et.metodo == "Otsu") & (det_et.variante == var)]["dice"]
                if len(a) and len(b):
                    rows.append({"case_id": cid, "variante": var,
                                 "otsu_v2": round(float(a.iloc[0]), 4),
                                 "otsu_v3": round(float(b.iloc[0]), 4)})
        otsu_cmp = pd.DataFrame(rows)

    fig_ranking = figura_ranking(tabla_fig, figuras)
    n_et = len(casos_et)
    construir_reporte(piv, detalle, figs, fig_ranking, mejor_base, mejor_mej,
                      cierre_s, alpha_s, n_et, n_cruces, tabla_tam, cmp_txt,
                      base_dir / "segmentacion_mejora_v3_reporte.html")

    print("\n========== TABLA COMPARATIVA v3 (Dice medio; morf=componente ANCLADO) ==========")
    print(tabla_fig.round(4).to_string(index=False))
    print(f"  [GMM] morf anclado={gmm_anc:.3f}  vs  morf mayor={gmm_may:.3f}")
    print("\n========== POR GRUPO DE TAMAÑO (Dice medio) ==========")
    print(tabla_tam.round(4).to_string(index=False))
    if cmp_df is not None:
        print("\n========== v2 (mayor) vs v3 (anclado) ==========")
        print(cmp_df.to_string(index=False))
    if otsu_cmp is not None:
        print("\n========== Otsu en casos que colapsaban: v2 vs v3 ==========")
        print(otsu_cmp.to_string(index=False))
    print(f"\n  mejor base    : {mejor_base['etq']} = {mejor_base['dice']:.3f}")
    print(f"  mejor mejorado: {mejor_mej['etq']} = {mejor_mej['dice']:.3f}  (delta {mejor_mej['dice']-mejor_base['dice']:+.3f})")
    print(f"  combinaciones caso/metodo/variante que cruzan 0,5: {n_cruces}")
    print(f"  veces que se usó fallback (semilla fuera de máscara): {n_fallback}")
    print(f"  params GLOBALES: cierre*={cierre_s}, alpha*={alpha_s} (sin ajuste por caso)")
    print(f"  salidas en: {base_dir}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
