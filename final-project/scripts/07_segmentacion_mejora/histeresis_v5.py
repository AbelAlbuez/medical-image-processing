#!/usr/bin/env python3
"""
Módulo 7 v5 — Histéresis (umbral doble) + Otsu 2-clases + ensamble con histéresis
=================================================================================
Tres estrategias nuevas, comparadas contra el mejor hasta ahora (ensamble union_post=0,370):

  A. RG_histeresis  : umbral doble sobre T1c. Núcleo = vóxeles >= alpha_alto*I_semilla;
                      crecimiento = vóxeles >= alpha_bajo*I_semilla SOLO si conectados al
                      núcleo (histéresis real, vía skimage.apply_hysteresis_threshold) +
                      componente anclado a la semilla. Barrido GLOBAL de (alto,bajo).
  B. Otsu_2clases   : Otsu n=2 (un umbral) en vez de n=3, clase alta + componente anclado.
  C. ensamble_hist  : unión de {Otsu anclado, RG, Watershed (v3)} + RG_histeresis, + post anclado.

Reglas anti-trampa: el GT solo ubica la semilla y evalúa; parámetros (alphas, n clases, cierre)
GLOBALES por promedio, nunca por caso. Solo SimpleITK/numpy/scikit-image.
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
from skimage.filters import apply_hysteresis_threshold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "04_segmentacion"))
sys.path.insert(0, str(ROOT / "07_segmentacion_mejora"))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import metricas as M             # noqa: E402
from comun import reporte as R              # noqa: E402
import seg_core as S                        # noqa: E402
from segmentacion_mejora import postproceso_anclado  # noqa: E402

import matplotlib.pyplot as plt             # noqa: E402

CIERRE_GLOBAL = 3
SWEEP_ALTO = [0.65, 0.75, 0.85]
SWEEP_BAJO = [0.45, 0.55]
REFERENCIA = {"ensamble_union_post (v4)": 0.370, "Otsu_anclado (v3)": 0.353}


def t1c_ref(cid):
    p = C.SALIDAS_MODULO["limpieza"] / "outputs" / f"{cid}-t1c_limpio.nii.gz"
    img = sitk.ReadImage(str(p), sitk.sitkFloat32)
    return img, io_zip.a_numpy(img)


def mascara_v3(cid, metodo):
    p = C.OUTPUT_DIR / "segmentacion_mejora" / "outputs" / f"{cid}-{metodo}-morf-v3-ET.nii.gz"
    return io_zip.a_numpy(sitk.ReadImage(str(p))).astype(np.uint8)


def intensidad_semilla(arr, seed):
    sx, sy, sz = int(seed[0]), int(seed[1]), int(seed[2])
    return float(arr[np.clip(sz, 0, arr.shape[0]-1), np.clip(sy, 0, arr.shape[1]-1),
                     np.clip(sx, 0, arr.shape[2]-1)])


def histeresis(arr, seed, a_alto, a_bajo):
    """
    Histéresis por umbral doble (núcleo brillante + crecimiento conectado al núcleo),
    devuelta como máscara cruda uint8 (antes del post-proceso anclado).
    """
    I = intensidad_semilla(arr, seed)
    low, high = a_bajo * I, a_alto * I
    if high <= low:
        high = low + 1e-6
    hyst = apply_hysteresis_threshold(arr, low, high)   # vóxeles >=low conectados a >=high
    return hyst.astype(np.uint8)


def nucleo_alto(arr, ref, seed, a_alto):
    """Solo el núcleo (>= alpha_alto*I) anclado a la semilla — para la figura."""
    I = intensidad_semilla(arr, seed)
    core = (arr >= a_alto * I).astype(np.uint8)
    m, _ = postproceso_anclado(core, ref, CIERRE_GLOBAL, seed)
    return m


# --------------------------------------------------------------------------- #
# Barrido global de (alpha_alto, alpha_bajo) para la histéresis
# --------------------------------------------------------------------------- #
def barrido_histeresis(casos_et, cache, outputs):
    filas = []
    for a, b in itertools.product(SWEEP_ALTO, SWEEP_BAJO):
        if a <= b:
            continue
        dices = []
        for cid in casos_et:
            ref, arr, seg, seed = cache[cid]
            h = histeresis(arr, seed, a, b)
            hp, _ = postproceso_anclado(h, ref, CIERRE_GLOBAL, seed)
            dices.append(M.dice(hp, M.mascara_et(seg)))
        filas.append({"alpha_alto": a, "alpha_bajo": b, "dice_medio": round(float(np.mean(dices)), 4)})
    df = pd.DataFrame(filas).sort_values("dice_medio", ascending=False)
    df.to_csv(outputs / "barrido_histeresis_v5.csv", index=False)
    mejor = df.iloc[0]
    return float(mejor["alpha_alto"]), float(mejor["alpha_bajo"]), df


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


def figura_histeresis(cid, arr, seg, nucleo, crecimiento, figuras):
    from matplotlib.colors import ListedColormap
    et = seg == C.LABEL_ET
    z = int(round(ndi.center_of_mass(et)[0])) if et.any() else arr.shape[0] // 2
    cpred = ListedColormap([(0, 0, 0, 0), (1.0, 0.3, 0.3, 0.55)])
    paneles = [("núcleo (umbral alto)", nucleo), ("histéresis (crecimiento bajo)", crecimiento)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax, (tt, mk) in zip(axes[:2], paneles):
        ax.imshow(_disp(arr[z]), cmap="gray", origin="lower", aspect="auto")
        ax.imshow(mk[z], cmap=cpred, origin="lower", aspect="auto")
        if et.any():
            ax.contour(et[z].astype(float), levels=[0.5], colors=["#22D3EE"], linewidths=0.8, origin="lower")
        ax.set_title(tt, fontsize=10); ax.axis("off")
    axes[2].imshow(_disp(arr[z]), cmap="gray", origin="lower", aspect="auto")
    if et.any():
        axes[2].contourf(et[z].astype(float), levels=[0.5, 1], colors=["#22D3EE"], alpha=0.4, origin="lower")
    axes[2].set_title("GT (ET)", fontsize=10); axes[2].axis("off")
    fig.suptitle(f"{cid} — histéresis (pred roja, GT cian)", fontsize=12); fig.tight_layout()
    return R.guardar_figura(fig, figuras / f"histeresis_v5_{cid}.png")


def figura_ranking(medias, figuras):
    nombres = list(medias.keys()) + list(REFERENCIA.keys())
    vals = list(medias.values()) + list(REFERENCIA.values())
    colores = [C.PALETA["azul"]] * len(medias) + [C.PALETA["suave"]] * len(REFERENCIA)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    barras = ax.bar(range(len(nombres)), vals, color=colores)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color=C.PALETA["rojo"], ls="--", lw=1, label="umbral 0,5")
    ax.set_xticks(range(len(nombres))); ax.set_xticklabels(nombres, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Dice medio (casos con ET)")
    ax.set_title("v5: histéresis / Otsu-2clases / ensamble vs mejor previo")
    ax.legend(fontsize=8); fig.tight_layout()
    return R.guardar_figura(fig, figuras / "ranking_v5.png")


def construir_reporte(tabla, tam, detalle, figs, fig_ranking, mejor, n_cruces, supera, ruta_html):
    kpis = [
        R.kpi(mejor["estrategia"], "mejor estrategia v5"),
        R.kpi(f"{mejor['dice']:.3f}", "Dice medio"),
        R.kpi(f"{mejor['dice']-0.370:+.3f}", "vs mejor previo (0,370)"),
        R.kpi(str(n_cruces), "casos que cruzan 0,5"),
    ]
    secciones = [
        R.seccion("Estrategias v5",
                  "<p><b>RG_histeresis</b>: umbral doble (núcleo brillante + crecimiento conectado), "
                  "anclado a la semilla. <b>Otsu_2clases</b>: Otsu n=2 anclado. <b>ensamble_hist</b>: "
                  "unión de las 3 máscaras v3 + la histéresis, con post-proceso anclado. Parámetros "
                  f"globales por promedio.</p><p>{supera}</p>"),
        R.seccion("Dice medio por estrategia (+ referencia)",
                  R.df_a_tabla_html(tabla.round(4)),
                  R.tarjeta_figura(R.png_a_base64(fig_ranking), "Estrategias v5 vs mejor previo; línea 0,5.")),
        R.seccion("Por grupo de tamaño (mejor estrategia)", R.df_a_tabla_html(tam.round(4))),
        R.seccion("Histéresis: núcleo vs crecimiento (caso bueno y malo)",
                  *[R.tarjeta_figura(R.png_a_base64(p), f"{cid}") for cid, p in figs.items()]),
        R.seccion("Detalle por caso y estrategia", R.df_a_tabla_html(detalle.round(4))),
    ]
    R.armar_reporte("Histéresis y ensamble de ET (v5) — T1c · BraTS 2024 GLI",
                    kpis, secciones,
                    subtitulo="Módulo 7 v5 · umbral doble + Otsu 2-clases + ensamble con histéresis",
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
    print(f"[v5] {len(casos)} casos ({len(casos_et)} con ET)\n")

    # Cache: ref, arr_t1c, seg, seed
    cache = {}
    for cid in casos:
        ref, arr = t1c_ref(cid)
        seg = io_zip.leer_seg_np(cid)
        seed = S.semilla_et_brillante(arr, seg) or S.semilla_fallback(arr)
        cache[cid] = (ref, arr, seg, seed)
        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    a_alto, a_bajo, df_sweep = barrido_histeresis(casos_et, cache, outputs)
    print(f"[v5] mejor histéresis: alpha_alto={a_alto}, alpha_bajo={a_bajo} "
          f"(Dice medio={df_sweep.iloc[0]['dice_medio']})\n")

    detalle_filas = []
    fuga_avisos = []
    figs_data = {}
    for i, cid in enumerate(casos, 1):
        ref, arr, seg, seed = cache[cid]
        et_presente = M.hay_et(seg)
        gt = M.mascara_et(seg)

        # A) histéresis
        h_raw = histeresis(arr, seed, a_alto, a_bajo)
        h_mask, _ = postproceso_anclado(h_raw, ref, CIERRE_GLOBAL, seed)
        # B) Otsu 2 clases
        o2, _info = S.segmentar_otsu(arr, clases=2, tomar_clase="alta", nombre="")
        o2_mask, _ = postproceso_anclado(o2, ref, CIERRE_GLOBAL, seed)
        # C) ensamble + histéresis (4 máscaras: otsu/rg/ws v3 + histéresis)
        v3 = [mascara_v3(cid, m) for m in ["otsu", "regiongrowing", "watershed"]]
        union4 = (np.sum(v3 + [h_mask], axis=0) >= 1).astype(np.uint8)
        ens_mask, _ = postproceso_anclado(union4, ref, CIERRE_GLOBAL, seed)

        cerebro = int((arr > 0).sum())
        for estr, mk in [("RG_histeresis", h_mask), ("Otsu_2clases", o2_mask),
                         ("ensamble_hist", ens_mask)]:
            vox = int(mk.sum())
            if vox > 0.5 * cerebro:                          # fuga: > 50% del cerebro
                fuga_avisos.append(f"{cid}/{estr}: {vox} vóxeles (>50% cerebro)")
            ev = M.evaluar_et(mk, seg)
            detalle_filas.append({
                "case_id": cid, "estrategia": estr, "et_presente": et_presente,
                "dice": ev["dice"], "jaccard": ev["jaccard"],
                "sensibilidad": ev["sensibilidad"], "especificidad": ev["especificidad"],
                "pred_voxeles": vox})
        figs_data[cid] = (arr, seg, nucleo_alto(arr, ref, seed, a_alto), h_mask)
        d_h = next(r["dice"] for r in detalle_filas if r["case_id"] == cid and r["estrategia"] == "RG_histeresis")
        print(f"  [{i}/{len(casos)}] {cid}  RG_histeresis Dice={'N/A' if d_h != d_h else round(d_h,3)}")
        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    detalle = pd.DataFrame(detalle_filas)
    detalle.to_csv(outputs / "metricas_v5.csv", index=False)
    det_et = detalle[detalle.et_presente].copy()

    medias = det_et.groupby("estrategia")["dice"].mean().to_dict()
    medias = {k: round(float(v), 4) for k, v in medias.items()}
    filas = [{"fila": k, "tipo": "v5", "dice": v} for k, v in medias.items()]
    for k, v in REFERENCIA.items():
        filas.append({"fila": k, "tipo": "referencia", "dice": v})
    tabla = pd.DataFrame(filas).sort_values("dice", ascending=False)
    tabla.to_csv(outputs / "tabla_comparativa_v5.csv", index=False)

    mejor_estr = max(medias, key=medias.get)
    mejor = {"estrategia": mejor_estr, "dice": medias[mejor_estr]}
    sens = float(det_et[det_et.estrategia == mejor_estr]["sensibilidad"].mean())
    espec = float(det_et[det_et.estrategia == mejor_estr]["especificidad"].mean())
    cruces = det_et[(det_et.estrategia == mejor_estr) & (det_et.dice >= 0.5)]
    n_cruces = int(len(cruces))
    prev = 0.370
    if mejor["dice"] > prev:
        supera = (f"La mejor estrategia v5 (<b>{mejor_estr}</b>, {mejor['dice']:.3f}) SUPERA al "
                  f"mejor previo (union_post 0,370) por {mejor['dice']-prev:+.3f}.")
    else:
        supera = (f"Ninguna estrategia v5 supera al mejor previo (union_post 0,370); la mejor es "
                  f"{mejor_estr} con {mejor['dice']:.3f} ({mejor['dice']-prev:+.3f}).")

    # por tamaño (mejor estrategia)
    eda = pd.read_csv(C.SALIDAS_MODULO["eda"] / "outputs" / "estadisticas_por_caso.csv")
    vol = {r.case_id: r.et_volumen_mm3 for r in eda.itertuples()}
    vols_et = np.array([vol[c] for c in casos_et], dtype=float)
    q1, q2 = np.percentile(vols_et, [33.33, 66.67])
    det_et["grupo"] = det_et["case_id"].map(
        lambda c: "pequeno" if vol.get(c, 0) <= q1 else ("mediano" if vol.get(c, 0) <= q2 else "grande"))
    tam = (det_et[det_et.estrategia == mejor_estr].groupby("grupo")["dice"].mean()
           .reindex(["pequeno", "mediano", "grande"]).reset_index())
    tam.columns = ["grupo", f"dice_{mejor_estr}"]
    tam.to_csv(outputs / "tabla_por_tamano_v5.csv", index=False)

    # figuras: 1 bueno, 1 malo según histéresis
    h = det_et[det_et.estrategia == "RG_histeresis"].sort_values("dice")
    cid_malo, cid_bueno = h.iloc[0]["case_id"], h.iloc[-1]["case_id"]
    figs = {}
    for cid in [cid_bueno, cid_malo]:
        arr, seg, nucleo, crec = figs_data[cid]
        figs[cid] = figura_histeresis(cid, arr, seg, nucleo, crec, figuras)
    fig_ranking = figura_ranking(medias, figuras)

    construir_reporte(tabla, tam, detalle, figs, fig_ranking, mejor, n_cruces, supera,
                      base_dir / "segmentacion_mejora_v5_reporte.html")

    print("\n========== TABLA COMPARATIVA v5 ==========")
    print(tabla.to_string(index=False))
    print(f"\n  mejor par histéresis: alto={a_alto}, bajo={a_bajo}  (Dice {df_sweep.iloc[0]['dice_medio']})")
    print(f"  {supera.replace('<b>','').replace('</b>','')}")
    print(f"  mejor estrategia: {mejor_estr}  Dice={mejor['dice']:.3f}  sens={sens:.3f}  espec={espec:.3f}")
    print(f"  casos que cruzan 0,5 con {mejor_estr}: {n_cruces}")
    print(f"  avisos de fuga (>50% cerebro): {len(fuga_avisos)}")
    for a in fuga_avisos:
        print("    -", a)
    print(f"  salidas en: {base_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()
