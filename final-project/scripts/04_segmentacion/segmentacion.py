#!/usr/bin/env python3
"""
Módulo 4 — SEGMENTACIÓN del Tumor Realzante (ET) sobre T1c LIMPIO
================================================================
Cuatro métodos clásicos (objetivo = seg==3, sobre T1c limpio del módulo 2):

  1. Otsu          — Otsu multinivel n=3, clase alta (avisa si histograma unimodal).
  2. RegionGrowing — ConfidenceConnected sembrado en el centroide del ET del GT
                     (semilla en ROJO en las figuras). MÉTODO ESTRELLA.
                     (ConnectedThreshold también está en seg_core como variante.)
  3. Watershed     — watershed marcador-controlado sobre el gradiente de T1c.
  4. GMM_multimodal— GMM sobre T1c+T2w+T2f (MULTIMODAL), método de COMPARACIÓN.

Casos: 8 con ET + 1 sin ET (trazabilidad en outputs/casos_segmentacion.json). Si falta
el T1c limpio de algún caso, se limpia con el módulo 2 (denoise→N4→percentil) ANTES de
segmentar, sin re-limpiar los ya existentes.

Métricas (metricas.py extendido): Dice, Jaccard, sensibilidad, especificidad. El caso
sin ET no calcula Dice/Jaccard/sensibilidad (N/A) pero verifica que cada método corre y
reporta los vóxeles predichos (falsos positivos).

Sweep de parámetros del region growing con itertools.product (--no_product = zip posicional).

Ejecución:
  python segmentacion.py
  python segmentacion.py --no_product
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_limpieza"))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import metricas as M             # noqa: E402
from comun import reporte as R              # noqa: E402
import seg_core as S                        # noqa: E402
import limpieza_core as LC                  # noqa: E402  (reutiliza el módulo 2)

import matplotlib.pyplot as plt             # noqa: E402

# Casos: 3 con ET ya conocidos + 5 nuevos con ET (rango de volumen) + 1 sin ET.
CASOS_CON_ET = [
    "BraTS-GLI-02063-106", "BraTS-GLI-00529-100", "BraTS-GLI-02062-100",
    "BraTS-GLI-02063-102", "BraTS-GLI-00498-100", "BraTS-GLI-00463-101",
    "BraTS-GLI-02060-100", "BraTS-GLI-00046-101",
]
CASO_SIN_ET = "BraTS-GLI-00005-100"

# Parámetros documentados de cada método.
OTSU_CLASES = 3
RG_ALPHA = 0.65         # umbral inferior = alpha * I_semilla (calibrado para ET en [0,1])
RG_CIERRE = 1           # radio del cierre morfológico
# Grid del sweep de region growing (itertools.product por defecto).
SWEEP_ALPHA = [0.55, 0.65, 0.75]
SWEEP_CIERRE = [1, 2, 3]

METODOS = ["Otsu", "RegionGrowing", "Watershed", "GMM_multimodal"]
MULTIMODALES = {"GMM_multimodal"}
NOMBRE_ARCHIVO = {"Otsu": "otsu", "RegionGrowing": "regiongrowing",
                  "Watershed": "watershed", "GMM_multimodal": "gmm_multimodal"}


# --------------------------------------------------------------------------- #
# Limpieza perezosa: asegura el T1c limpio (reutiliza módulo 2)
# --------------------------------------------------------------------------- #
def asegurar_t1c_limpio(cid: str):
    """Devuelve la imagen SITK del T1c limpio; lo genera (y guarda) si no existe."""
    dst = C.SALIDAS_MODULO["limpieza"] / "outputs" / f"{cid}-t1c_limpio.nii.gz"
    if dst.exists():
        import SimpleITK as sitk
        return sitk.ReadImage(str(dst), sitk.sitkFloat32)
    print(f"      (limpieza faltante de T1c) generando {dst.name} ...")
    img = io_zip.leer_sitk(cid, "t1c")
    limpio = LC.limpiar_t1c(img)
    io_zip.guardar_sitk(limpio, dst)
    return limpio


def limpiar_modalidad_memoria(cid: str, mod: str):
    """Limpia una modalidad (t2w/t2f) en memoria para el GMM multimodal (no la guarda)."""
    img = io_zip.leer_sitk(cid, mod)
    if img is None:
        return None
    return io_zip.a_numpy(LC.limpiar_t1c(img))


# --------------------------------------------------------------------------- #
# Segmentación de un caso (4 métodos)
# --------------------------------------------------------------------------- #
def segmentar_caso(cid: str):
    """Devuelve (preds, ref_img, seg, seeds{metodo:seed}, seed_fb, info_otsu)."""
    ref = asegurar_t1c_limpio(cid)            # sitk (geometría del original)
    arr_t1c = io_zip.a_numpy(ref)
    seg = io_zip.leer_seg_np(cid)
    hay_et = S.centroide_et_index(seg) is not None

    # Semillas: region growing y watershed en el ET MÁS BRILLANTE (el centroide del ET
    # cae en la cavidad oscura en ET post-tratamiento y arruina ambos métodos sembrados).
    seed_rg = S.semilla_et_brillante(arr_t1c, seg)
    seed_ws = seed_rg
    seed_fb = not hay_et
    if seed_fb:
        seed_rg = S.semilla_fallback(arr_t1c)   # garantiza que el método corre sin ET
        seed_ws = None                          # sin ET, watershed deriva el marcador de Otsu

    # 1) Otsu
    m_otsu, info_otsu = S.segmentar_otsu(arr_t1c, clases=OTSU_CLASES,
                                         tomar_clase="alta", nombre=f"{cid}/t1c")
    # 2) RegionGrowing (ventana brillante desde el ET más brillante). MÉTODO ESTRELLA.
    m_rg = S.connected_threshold(ref, seed_rg, alpha=RG_ALPHA, radio_cierre=RG_CIERRE)
    # 3) Watershed (marcador de tumor en el ET brillante; si no hay ET, clase alta de Otsu)
    m_ws = S.segmentar_watershed(arr_t1c, seed=seed_ws)
    # 4) GMM multimodal (T1c + T2w + T2f limpios)
    arr_t2w = limpiar_modalidad_memoria(cid, "t2w")
    arr_t2f = limpiar_modalidad_memoria(cid, "t2f")
    vols = [arr_t1c] + [a for a in (arr_t2w, arr_t2f) if a is not None]
    m_gmm = S.segmentar_clustering(vols, metodo="gmm", n_clusters=4, idx_referencia=0)

    preds = {"Otsu": m_otsu, "RegionGrowing": m_rg, "Watershed": m_ws, "GMM_multimodal": m_gmm}
    seeds = {"Otsu": None, "RegionGrowing": seed_rg,
             "Watershed": seed_ws if seed_ws is not None else seed_rg,
             "GMM_multimodal": None}
    return preds, ref, seg, seeds, seed_fb, info_otsu


# --------------------------------------------------------------------------- #
# Figura por método+caso
# --------------------------------------------------------------------------- #
def figura_metodo(cid, metodo, arr_t1c, seg, pred, seed, seed_fallback, figuras):
    et = seg == C.LABEL_ET
    z = int(round(ndi.center_of_mass(et)[0])) if et.any() else arr_t1c.shape[0] // 2
    bg = arr_t1c[z]
    usa_semilla = seed is not None
    if usa_semilla:
        sx, sy, sz = seed

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    # Panel 1: T1c original (limpio)
    axes[0].imshow(bg, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("T1c limpio", fontsize=10); axes[0].axis("off")
    # Panel 2: GT (ET en contorno cian)
    axes[1].imshow(bg, cmap="gray", origin="lower", aspect="auto")
    if et.any():
        axes[1].contour(et[z].astype(float), levels=[0.5], colors=["#22D3EE"],
                        linewidths=1.0, origin="lower")
    axes[1].set_title("GT — ET", fontsize=10); axes[1].axis("off")
    # Panel 3: predicción (rojo) sobre T1c + contorno GT
    axes[2].imshow(bg, cmap="gray", origin="lower", aspect="auto")
    from matplotlib.colors import ListedColormap
    cmap_pred = ListedColormap([(0, 0, 0, 0), (1.0, 0.2, 0.2, 0.5)])
    axes[2].imshow(pred[z], cmap=cmap_pred, origin="lower", aspect="auto")
    if et.any():
        axes[2].contour(et[z].astype(float), levels=[0.5], colors=["#22D3EE"],
                        linewidths=0.8, origin="lower")
    axes[2].set_title("Predicción (roja) + GT", fontsize=10); axes[2].axis("off")
    # Semilla en ROJO donde aplique y caiga en este corte
    if usa_semilla and abs(sz - z) <= 2:
        etiqueta_semilla = "semilla (auto)" if seed_fallback else "semilla ET"
        for ax in (axes[0], axes[2]):
            ax.plot(sx, sy, "o", color=C.PALETA["rojo"], markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2)
        axes[0].text(0.02, 0.02, etiqueta_semilla, color=C.PALETA["rojo"],
                     transform=axes[0].transAxes, fontsize=8)

    fig.suptitle(f"{cid} — {metodo} (corte z={z})", fontsize=12)
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / f"seg_{NOMBRE_ARCHIVO[metodo]}_{cid}.png")


def figura_ranking(resumen, figuras):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colores = [C.PALETA["morado"] if mm else C.PALETA["azul"]
               for mm in resumen["multimodal"]]
    barras = ax.bar(resumen["metodo"], resumen["dice"], color=colores)
    for b, v in zip(barras, resumen["dice"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Dice medio (casos con ET)")
    ax.set_title("Ranking de métodos por Dice (ET)")
    ax.set_ylim(0, max(0.05, float(resumen["dice"].max()) * 1.2))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C.PALETA["azul"], label="mono-modal (T1c)"),
                       Patch(color=C.PALETA["morado"], label="multimodal")], fontsize=8)
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / "ranking_metodos.png")


# --------------------------------------------------------------------------- #
# Sweep de region growing (itertools.product / zip)
# --------------------------------------------------------------------------- #
def correr_sweep(casos_et, usar_product: bool, outputs: Path):
    if usar_product:
        combos = list(itertools.product(SWEEP_ALPHA, SWEEP_CIERRE))
        modo = "product"
    else:
        combos = list(zip(SWEEP_ALPHA, SWEEP_CIERRE))
        modo = "zip (--no_product)"
    print(f"\n[SWEEP] region growing (alpha × cierre) — {modo}: {len(combos)} combinaciones "
          f"sobre {len(casos_et)} casos con ET")
    filas = []
    for cid in casos_et:
        ref = asegurar_t1c_limpio(cid)
        arr = io_zip.a_numpy(ref)
        seg = io_zip.leer_seg_np(cid)
        seed = S.semilla_et_brillante(arr, seg)
        if seed is None:
            continue
        gt = M.mascara_et(seg)
        for alpha, cierre in combos:
            mask = S.connected_threshold(ref, seed, alpha=alpha, radio_cierre=cierre)
            filas.append({"case_id": cid, "alpha": alpha, "cierre": cierre,
                          "dice": round(M.dice(mask, gt), 4)})
    df = pd.DataFrame(filas)
    df.to_csv(outputs / "sweep_region_growing.csv", index=False)
    if len(df):
        mejor = (df.groupby(["alpha", "cierre"])["dice"].mean()
                   .sort_values(ascending=False).head(1))
        print(f"[SWEEP] mejor combinación media: {mejor.to_dict()}")
    return df, modo


# --------------------------------------------------------------------------- #
# Tabla por caso con semáforo (HTML)
# --------------------------------------------------------------------------- #
def tabla_semaforo_html(detalle: pd.DataFrame, metrica: str = "dice") -> str:
    pivot = detalle.pivot_table(index="case_id", columns="metodo", values=metrica,
                                aggfunc="first")
    pivot = pivot.reindex(columns=METODOS)
    filas_html = ["<table class='tabla'><thead><tr><th>case_id</th>" +
                  "".join(f"<th>{m}</th>" for m in METODOS) + "</tr></thead><tbody>"]
    for cid, row in pivot.iterrows():
        celdas = [f"<td style='text-align:left'>{cid}</td>"]
        for m in METODOS:
            v = row[m]
            if pd.isna(v):
                bg, txt = "#e5e7eb", "N/A"
            elif v >= 0.80:
                bg, txt = "#c6efce", f"{v:.3f}"
            elif v >= 0.50:
                bg, txt = "#ffeb9c", f"{v:.3f}"
            else:
                bg, txt = "#ffc7ce", f"{v:.3f}"
            celdas.append(f"<td style='background:{bg};text-align:center'>{txt}</td>")
        filas_html.append("<tr>" + "".join(celdas) + "</tr>")
    filas_html.append("</tbody></table>")
    return "\n".join(filas_html)


# --------------------------------------------------------------------------- #
# Reporte HTML
# --------------------------------------------------------------------------- #
def construir_reporte(detalle, resumen, figs, fig_ranking, n_et, ruta_html):
    mejor = resumen.iloc[0]
    kpis = [
        R.kpi(str(mejor["metodo"]), "mejor método (por Dice)"),
        R.kpi(f"{mejor['dice']:.3f}", "Dice del mejor método"),
        R.kpi(str(n_et), "casos con ET (promedio)"),
        R.kpi("T1c", "modalidad objetivo (ET)"),
    ]
    secciones = []
    secciones.append(R.seccion(
        "Objetivo y métodos",
        "<p>Segmentación del <b>Tumor Realzante (ET, seg==3)</b> sobre <b>T1c limpio</b>. "
        "Cuatro métodos clásicos: <b>Otsu</b> (multinivel n=3, clase alta), "
        "<b>RegionGrowing</b> (ConfidenceConnected sembrado en el centroide del ET — método "
        "estrella), <b>Watershed</b> (marcador-controlado) y <b>GMM_multimodal</b> "
        "(T1c+T2w+T2f, comparación). Métricas: Dice, Jaccard, sensibilidad, especificidad.</p>"))
    secciones.append(R.seccion(
        "Resumen por método (promedio sobre casos con ET)",
        R.df_a_tabla_html(resumen.round(4)),
        R.tarjeta_figura(R.png_a_base64(fig_ranking), "Dice medio por método (morado = multimodal).")))
    secciones.append(R.seccion(
        "Dice por caso (semáforo: verde ≥0.80 · amarillo 0.50–0.80 · rojo <0.50 · N/A sin ET)",
        tabla_semaforo_html(detalle, "dice")))
    # Sección dedicada GMM multimodal vs mono-modales
    mono = resumen[~resumen["multimodal"]]
    multi = resumen[resumen["multimodal"]]
    txt_cmp = (f"<p>El método <b>multimodal (GMM, T1c+T2w+T2f)</b> obtiene Dice medio "
               f"<b>{float(multi['dice'].iloc[0]):.3f}</b>, frente al mejor mono-modal "
               f"(<b>{mono.iloc[0]['metodo']}</b>) con <b>{float(mono.iloc[0]['dice']):.3f}</b>. "
               f"El multimodal usa varias modalidades a la vez; los mono-modales operan solo "
               f"sobre T1c, que es la modalidad donde el realce (ET) es más separable (EDA).</p>")
    secciones.append(R.seccion("GMM multimodal vs métodos mono-modales", txt_cmp,
                               R.df_a_tabla_html(resumen[["metodo", "multimodal", "dice",
                                                          "jaccard", "sensibilidad",
                                                          "especificidad"]].round(4))))
    # Galería curada de figuras
    bloques = [R.tarjeta_figura(R.png_a_base64(p), Path(p).name) for p in figs]
    secciones.append(R.seccion("Figuras (casos demostrativos × métodos)", *bloques))
    R.armar_reporte(
        "Segmentación de ET — T1c · BraTS 2024 GLI",
        kpis, secciones,
        subtitulo="Módulo 4 · Otsu · RegionGrowing · Watershed · GMM multimodal",
        ruta_salida=ruta_html)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Segmentación clásica de ET sobre T1c.")
    ap.add_argument("--no_product", action="store_true",
                    help="Sweep con emparejado posicional (zip) en vez de itertools.product.")
    args = ap.parse_args()

    dirs = C.dirs_modulo("segmentacion")
    casos = CASOS_CON_ET + [CASO_SIN_ET]

    # Trazabilidad de la selección.
    with open(dirs["outputs"] / "casos_segmentacion.json", "w", encoding="utf-8") as f:
        json.dump({"con_et": CASOS_CON_ET, "sin_et": [CASO_SIN_ET], "todos": casos,
                   "n_con_et": len(CASOS_CON_ET)}, f, ensure_ascii=False, indent=2)
    print(f"[SEG] {len(casos)} casos ({len(CASOS_CON_ET)} con ET + 1 sin ET)\n")

    detalle_filas = []
    figs_demostrativas = []
    demostrativos = {"BraTS-GLI-00529-100", "BraTS-GLI-02063-102",
                     "BraTS-GLI-02060-100", CASO_SIN_ET}
    for i, cid in enumerate(casos, 1):
        print(f"  [{i}/{len(casos)}] {cid}")
        preds, ref, seg, seeds, seed_fb, info_otsu = segmentar_caso(cid)
        arr_t1c = io_zip.a_numpy(ref)
        et_presente = M.hay_et(seg)

        for metodo in METODOS:
            pred = preds[metodo]
            # Guardar máscara .nii.gz preservando geometría
            dst = dirs["outputs"] / f"{cid}-{NOMBRE_ARCHIVO[metodo]}-ET.nii.gz"
            io_zip.guardar_sitk(io_zip.desde_numpy(pred.astype(np.uint8), ref), dst)
            # Métricas ET
            ev = M.evaluar_et(pred, seg)
            detalle_filas.append({
                "case_id": cid, "metodo": metodo, "multimodal": metodo in MULTIMODALES,
                "et_presente": et_presente,
                "dice": ev["dice"], "jaccard": ev["jaccard"],
                "sensibilidad": ev["sensibilidad"], "especificidad": ev["especificidad"],
                "pred_voxeles": int(pred.sum()),
            })
            # Figura (todas a disco; en HTML solo las de casos demostrativos)
            p = figura_metodo(cid, metodo, arr_t1c, seg, pred, seeds[metodo], seed_fb, dirs["figuras"])
            if cid in demostrativos:
                figs_demostrativas.append(p)
            d = ev["dice"]
            d_txt = "N/A" if (d != d) else f"{d:.3f}"
            print(f"        {metodo:14s} Dice={d_txt:>6s}  pred_vox={int(pred.sum()):>7d}")

        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    detalle = pd.DataFrame(detalle_filas)
    detalle.to_csv(dirs["outputs"] / "metricas_segmentacion.csv", index=False)

    resumen = M.resumen_por_metodo(detalle)
    # Asegurar orden de columnas legible
    resumen = resumen[["metodo", "multimodal", "dice", "jaccard", "sensibilidad", "especificidad"]]
    resumen.to_csv(dirs["outputs"] / "metricas_resumen.csv", index=False)

    # Sweep de region growing
    df_sweep, modo_sweep = correr_sweep(CASOS_CON_ET, not args.no_product, dirs["outputs"])

    # Figuras agregadas + reporte
    fig_ranking = figura_ranking(resumen, dirs["figuras"])
    n_et = int(detalle[detalle["et_presente"]]["case_id"].nunique())
    print("\n[SEG] generando reporte HTML...")
    construir_reporte(detalle, resumen, figs_demostrativas, fig_ranking, n_et, dirs["reporte"])

    print("\n========== RESUMEN SEGMENTACIÓN ==========")
    print(f"  casos con ET en promedio: {n_et}")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(resumen.round(4).to_string(index=False))
    print(f"\n  sweep: {modo_sweep}  ({len(df_sweep)} filas)")
    print(f"  salidas en: {dirs['base']}")
    print("==========================================")


if __name__ == "__main__":
    main()
