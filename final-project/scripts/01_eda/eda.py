#!/usr/bin/env python3
"""
Módulo 1 — EDA (Análisis Exploratorio) orientado a ET/T1c
=========================================================
Recorre hasta MAX_CASOS del ZIP de TrainingData (lectura DIRECTA desde el ZIP) y
caracteriza el problema con un único NORTE: segmentar el Tumor Realzante (ET, label 3)
sobre T1c con métodos clásicos.

Produce (en output/eda/):
  outputs/parametros_eda.json        -> parámetros para los módulos siguientes
  outputs/estadisticas_por_caso.csv  -> una fila por caso
  outputs/separabilidad.csv          -> separabilidad ET vs sano por modalidad
  figuras/*.png                      -> figuras sueltas
  eda_reporte.html                   -> reporte autocontenido

Ejecución:
  python eda.py                 # MAX_CASOS por defecto (100)
  python eda.py --max_casos 30  # subconjunto menor para pruebas rápidas

NO descomprime el ZIP completo: extrae puntualmente caso+modalidad a un temporal y lo
borra tras procesar cada caso para no llenar el disco.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu

# Hacer importable el paquete `comun` (scripts/ está un nivel arriba de 01_eda/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import reporte as R              # noqa: E402

import matplotlib.pyplot as plt             # noqa: E402


# --------------------------------------------------------------------------- #
# Cálculos por caso
# --------------------------------------------------------------------------- #
def _bhattacharyya_1d(mu1, v1, mu2, v2) -> float:
    """Distancia de Bhattacharyya entre dos gaussianas 1D (mayor = más separables)."""
    v1 = max(v1, 1e-8); v2 = max(v2, 1e-8)
    term_var = 0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2.0))
    term_media = 0.25 * ((mu1 - mu2) ** 2) / (v1 + v2)
    return float(term_var + term_media)


def _fisher_ratio(mu1, v1, mu2, v2) -> float:
    """Fisher Discriminant Ratio: (mu1-mu2)^2 / (var1+var2). Mayor = más separable."""
    return float((mu1 - mu2) ** 2 / max(v1 + v2, 1e-8))


def analizar_caso(case_id: str, n_muestras_hist: int = 5000):
    """
    Calcula estadísticas de un caso. Devuelve (fila_dict, muestras_hist) donde
    muestras_hist = {mod: ndarray normalizado a p99 dentro del cerebro} para los
    histogramas agregados. Devuelve (None, None) si falta T1c o seg.
    """
    seg = io_zip.leer_seg_np(case_id)
    img_t1c = io_zip.leer_sitk(case_id, "t1c")
    if seg is None or img_t1c is None:
        return None, None

    arr_t1c = io_zip.a_numpy(img_t1c)
    spacing = img_t1c.GetSpacing()                       # (sx, sy, sz)
    vox_vol = float(np.prod(spacing))                    # mm^3 por vóxel
    iso_1mm = io_zip.verificar_geometria(img_t1c, case_id)

    cerebro = arr_t1c > 0
    et = seg == C.LABEL_ET
    et_presente = bool(et.any())
    et_voxeles = int(et.sum())

    # Otsu multinivel n=3 sobre T1c dentro del cerebro (umbral bajo/alto crudos).
    vals_t1c = arr_t1c[cerebro]
    otsu_bajo = otsu_alto = float("nan")
    if vals_t1c.size > 3 and np.unique(vals_t1c).size >= 3:
        try:
            t = threshold_multiotsu(vals_t1c, classes=3)
            otsu_bajo, otsu_alto = float(t[0]), float(t[1])
        except Exception:
            pass

    fila = {
        "case_id": case_id,
        "shape": "x".join(str(s) for s in arr_t1c.shape),     # (z,y,x)
        "spacing": "x".join(f"{s:.2f}" for s in spacing),
        "iso_1mm": bool(iso_1mm),
        "et_presente": et_presente,
        "et_voxeles": et_voxeles,
        "et_volumen_mm3": round(et_voxeles * vox_vol, 1),
        "otsu_t1c_bajo": round(otsu_bajo, 1) if otsu_bajo == otsu_bajo else np.nan,
        "otsu_t1c_alto": round(otsu_alto, 1) if otsu_alto == otsu_alto else np.nan,
        "t1c_media_cerebro": round(float(vals_t1c.mean()), 1) if vals_t1c.size else np.nan,
        "t1c_p99_cerebro": round(float(np.percentile(vals_t1c, 99)), 1) if vals_t1c.size else np.nan,
    }

    # Separabilidad ET vs tejido sano por modalidad (solo si hay ET).
    muestras = {}
    rng = np.random.default_rng(C.SEED)
    for mod in C.MODALIDADES_IMG:
        arr = arr_t1c if mod == "t1c" else io_zip.leer_np(case_id, mod)
        if arr is None:
            fila[f"fdr_{mod}"] = np.nan
            fila[f"bhatt_{mod}"] = np.nan
            continue
        m_cerebro = arr > 0
        # Histograma agregado: muestra de vóxeles de cerebro normalizados a su p99.
        vc = arr[m_cerebro]
        if vc.size:
            p99 = np.percentile(vc, 99) or 1.0
            muestra = vc / (p99 if p99 > 0 else 1.0)
            if muestra.size > n_muestras_hist:
                muestra = rng.choice(muestra, n_muestras_hist, replace=False)
            muestras[mod] = muestra.astype(np.float32)
        # Separabilidad ET vs sano (sano = cerebro & seg==0).
        if et_presente:
            sano = m_cerebro & (seg == 0)
            v_et = arr[et]
            v_sa = arr[sano]
            if v_et.size > 1 and v_sa.size > 1:
                mu_e, var_e = float(v_et.mean()), float(v_et.var())
                mu_s, var_s = float(v_sa.mean()), float(v_sa.var())
                fila[f"fdr_{mod}"] = round(_fisher_ratio(mu_e, var_e, mu_s, var_s), 4)
                fila[f"bhatt_{mod}"] = round(_bhattacharyya_1d(mu_e, var_e, mu_s, var_s), 4)
            else:
                fila[f"fdr_{mod}"] = np.nan
                fila[f"bhatt_{mod}"] = np.nan
        else:
            fila[f"fdr_{mod}"] = np.nan
            fila[f"bhatt_{mod}"] = np.nan

    return fila, muestras


# --------------------------------------------------------------------------- #
# Selección de casos demostrativos
# --------------------------------------------------------------------------- #
def elegir_demostrativos(df: pd.DataFrame) -> dict:
    """
    Elige casos demostrativos: ET grande / mediano / pequeño (por volumen, robusto a
    percentiles) + 1 SIN ET (robustez). Devuelve dict {categoria: case_id}.
    """
    con_et = df[df["et_presente"]].sort_values("et_volumen_mm3")
    elegidos = {}
    if len(con_et):
        vols = con_et["et_volumen_mm3"].to_numpy()
        for cat, pct in [("pequeno", 15), ("mediano", 50), ("grande", 85)]:
            objetivo = np.percentile(vols, pct)
            idx = int(np.argmin(np.abs(vols - objetivo)))
            elegidos[cat] = str(con_et.iloc[idx]["case_id"])
    # Evitar repetidos si hay muy pocos casos con ET.
    vistos = set()
    for cat in ["pequeno", "mediano", "grande"]:
        if cat in elegidos and elegidos[cat] in vistos:
            # buscar el siguiente caso con ET no usado
            restantes = [c for c in con_et["case_id"] if c not in vistos]
            if restantes:
                elegidos[cat] = str(restantes[len(vistos) % len(restantes)])
        vistos.add(elegidos.get(cat))
    sin_et = df[~df["et_presente"]]
    if len(sin_et):
        elegidos["sin_et"] = str(sin_et.iloc[0]["case_id"])
    return elegidos


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def fig_histogramas(hist_acum: dict, figuras: Path) -> Path:
    """Histogramas agregados de intensidad (normalizada a p99) por modalidad."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colores = {"t1n": C.PALETA["suave"], "t1c": C.PALETA["azul"],
               "t2w": C.PALETA["naranja"], "t2f": C.PALETA["morado"]}
    for mod in C.MODALIDADES_IMG:
        datos = hist_acum.get(mod)
        if datos is None or len(datos) == 0:
            continue
        ax.hist(datos, bins=120, range=(0, 1.5), histtype="step", linewidth=1.8,
                color=colores[mod], label=mod, density=True)
    ax.set_xlabel("Intensidad normalizada a p99 (dentro del cerebro)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de intensidades por modalidad (agregado)")
    ax.legend()
    return R.guardar_figura(fig, figuras / "eda_histogramas_modalidades.png")


def fig_separabilidad(sep_df: pd.DataFrame, figuras: Path) -> Path:
    """Barras de FDR y Bhattacharyya medianos por modalidad (resalta T1c para ET)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mods = sep_df["modalidad"].tolist()
    colores = [C.PALETA["azul"] if m == "t1c" else C.PALETA["suave"] for m in mods]
    for ax, col, titulo in [(axes[0], "fdr_mediana", "Fisher Discriminant Ratio (mediana)"),
                            (axes[1], "bhatt_mediana", "Distancia de Bhattacharyya (mediana)")]:
        ax.bar(mods, sep_df[col], color=colores)
        ax.set_title(titulo)
        ax.set_ylabel(col)
        for i, v in enumerate(sep_df[col]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Separabilidad ET vs tejido sano (mayor = mejor para segmentar ET)")
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / "eda_separabilidad.png")


def fig_volumen_et(df: pd.DataFrame, figuras: Path) -> Path:
    """Distribución del volumen de ET entre casos (variabilidad inter-caso)."""
    vols = df[df["et_presente"]]["et_volumen_mm3"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if vols.size:
        ax.hist(vols, bins=30, color=C.PALETA["verde"], alpha=0.85, edgecolor="white")
        ax.axvline(np.median(vols), color=C.PALETA["rojo"], linestyle="--",
                   label=f"mediana={np.median(vols):.0f} mm³")
        ax.legend()
    ax.set_xlabel("Volumen de ET (mm³)")
    ax.set_ylabel("Nº de casos")
    ax.set_title("Variabilidad inter-caso del volumen de ET")
    return R.guardar_figura(fig, figuras / "eda_volumen_et.png")


def fig_otsu_t1c(df: pd.DataFrame, figuras: Path) -> Path:
    """Distribución del umbral alto de Otsu n=3 en T1c (referencia ~1450)."""
    vals = df["otsu_t1c_alto"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if vals.size:
        ax.hist(vals, bins=30, color=C.PALETA["azul"], alpha=0.85, edgecolor="white")
        ax.axvline(np.median(vals), color=C.PALETA["rojo"], linestyle="--",
                   label=f"mediana={np.median(vals):.0f}")
    ax.set_xlabel("Umbral alto de Otsu n=3 en T1c (escala cruda)")
    ax.set_ylabel("Nº de casos")
    ax.set_title("Umbral de Otsu en T1c por caso")
    ax.legend()
    return R.guardar_figura(fig, figuras / "eda_otsu_t1c.png")


def fig_casos_demostrativos(elegidos: dict, figuras: Path) -> Path:
    """
    Panel por caso demostrativo: T1c (imagen ORIGINAL) en el corte axial del centroide
    del ET, con el ET superpuesto en ROJO. El caso sin ET muestra el corte central.
    """
    cats = [c for c in ["grande", "mediano", "pequeno", "sin_et"] if c in elegidos]
    fig, axes = plt.subplots(1, len(cats), figsize=(3.6 * len(cats), 3.9))
    if len(cats) == 1:
        axes = [axes]
    from scipy import ndimage as ndi
    for ax, cat in zip(axes, cats):
        cid = elegidos[cat]
        t1c = io_zip.leer_np(cid, "t1c")
        seg = io_zip.leer_seg_np(cid)
        et = (seg == C.LABEL_ET)
        if et.any():
            z = int(round(ndi.center_of_mass(et)[0]))
        else:
            z = t1c.shape[0] // 2
        sl = t1c[z]
        m = sl > 0
        if m.any():
            lo, hi = np.percentile(sl[m], [1, 99])
            disp = np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)
        else:
            disp = sl
        ax.imshow(disp, cmap="gray", origin="lower", aspect="auto")
        if et.any():
            ax.contour(et[z].astype(float), levels=[0.5], colors=[C.PALETA["rojo"]],
                       linewidths=1.2, origin="lower")
        ax.set_title(f"{cat}\n{cid}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Casos demostrativos — T1c original + ET (rojo)")
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / "eda_casos_demostrativos.png")


# --------------------------------------------------------------------------- #
# Reporte HTML
# --------------------------------------------------------------------------- #
def construir_reporte(params, df, sep_df, figs, ruta_html):
    kpis = [
        R.kpi(str(params["n_casos_procesados"]), "casos procesados"),
        R.kpi(f'{params["casos_con_et"]}', "casos con ET"),
        R.kpi(f'{params["otsu_t1c_alto_mediana"]:.0f}', "Otsu T1c (mediana)"),
        R.kpi(params["mejor_modalidad_et"].upper(), "mejor modalidad p/ ET"),
    ]
    secciones = []
    secciones.append(R.seccion(
        "Objetivo y alcance",
        f"<p>Objetivo: segmentar el <b>Tumor Realzante (ET, label {C.LABEL_ET})</b> sobre "
        f"<b>{params['modalidad'].upper()}</b> con métodos clásicos. Este EDA caracteriza "
        f"{params['n_casos_procesados']} casos del ZIP de entrenamiento.</p>"
        f"<p>De ellos, <b>{params['casos_con_et']}</b> contienen ET (seg==3). La separabilidad "
        f"confirma que <b>{params['mejor_modalidad_et'].upper()}</b> es la mejor modalidad para ET.</p>"))
    secciones.append(R.seccion(
        "Separabilidad ET vs tejido sano por modalidad",
        R.tarjeta_figura(R.png_a_base64(figs["separabilidad"]),
                         "FDR y Bhattacharyya medianos por modalidad."),
        R.df_a_tabla_html(sep_df)))
    secciones.append(R.seccion(
        "Distribución de intensidades",
        R.tarjeta_figura(R.png_a_base64(figs["histogramas"]),
                         "Intensidad normalizada a p99 dentro del cerebro, agregada sobre los casos.")))
    secciones.append(R.seccion(
        "Umbral de Otsu en T1c",
        R.tarjeta_figura(R.png_a_base64(figs["otsu"]),
                         f"Umbral alto de Otsu n=3 en T1c. Mediana={params['otsu_t1c_alto_mediana']:.0f} "
                         f"(referencia ~1450).")))
    secciones.append(R.seccion(
        "Volumen de ET y variabilidad inter-caso",
        R.tarjeta_figura(R.png_a_base64(figs["volumen"]),
                         "Distribución del volumen de ET (mm³) entre casos con ET.")))
    secciones.append(R.seccion(
        "Casos demostrativos seleccionados",
        R.tarjeta_figura(R.png_a_base64(figs["demostrativos"]),
                         "ET grande / mediano / pequeño + 1 sin ET (robustez)."),
        R.df_a_tabla_html(pd.DataFrame(
            [{"categoria": k, "case_id": v} for k, v in params["casos_demostrativos_detalle"].items()]))))
    secciones.append(R.seccion(
        "Estadísticas por caso (muestra)",
        R.df_a_tabla_html(df.head(15))))
    R.armar_reporte(
        "EDA — BraTS 2024 GLI · Tumor Realzante (ET) sobre T1c",
        kpis, secciones,
        subtitulo="Módulo 1 · Análisis exploratorio orientado a segmentación clásica de ET",
        ruta_salida=ruta_html)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="EDA BraTS 2024 GLI orientado a ET/T1c.")
    ap.add_argument("--max_casos", type=int, default=C.MAX_CASOS_DEFAULT)
    args = ap.parse_args()

    dirs = C.dirs_modulo("eda")
    print(f"[EDA] ZIP: {C.ZIP_TRAINING.name}")
    casos = io_zip.listar_casos(C.ZIP_TRAINING, max_casos=args.max_casos)
    print(f"[EDA] casos a procesar: {len(casos)} (max_casos={args.max_casos})\n")

    filas = []
    hist_acum = {m: [] for m in C.MODALIDADES_IMG}
    for i, cid in enumerate(casos, 1):
        fila, muestras = analizar_caso(cid)
        if fila is None:
            print(f"  [{i:3d}/{len(casos)}] {cid}  -> SIN T1c/seg, omitido")
            continue
        filas.append(fila)
        if muestras:
            for m, arr in muestras.items():
                hist_acum[m].append(arr)
        marca = "ET" if fila["et_presente"] else "--"
        print(f"  [{i:3d}/{len(casos)}] {cid}  {marca}  "
              f"vol_ET={fila['et_volumen_mm3']:>9.0f} mm³  otsu_alto={fila['otsu_t1c_alto']}")
        # Liberar disco: borrar los temporales de este caso.
        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    if not filas:
        print("\n[ERROR] No se procesó ningún caso. Revisa el ZIP y las rutas.")
        sys.exit(1)

    df = pd.DataFrame(filas)
    df.to_csv(dirs["outputs"] / "estadisticas_por_caso.csv", index=False)

    # Separabilidad agregada por modalidad.
    sep_rows = []
    for mod in C.MODALIDADES_IMG:
        fdr = df[f"fdr_{mod}"].dropna()
        bha = df[f"bhatt_{mod}"].dropna()
        sep_rows.append({
            "modalidad": mod,
            "fdr_mediana": round(float(fdr.median()), 4) if len(fdr) else np.nan,
            "fdr_media": round(float(fdr.mean()), 4) if len(fdr) else np.nan,
            "bhatt_mediana": round(float(bha.median()), 4) if len(bha) else np.nan,
            "bhatt_media": round(float(bha.mean()), 4) if len(bha) else np.nan,
            "n_casos": int(len(fdr)),
        })
    sep_df = pd.DataFrame(sep_rows).sort_values("fdr_mediana", ascending=False)
    sep_df.to_csv(dirs["outputs"] / "separabilidad.csv", index=False)
    mejor_mod = str(sep_df.iloc[0]["modalidad"])

    # Casos demostrativos.
    elegidos = elegir_demostrativos(df)

    con_et = df[df["et_presente"]]
    otsu_alto = df["otsu_t1c_alto"].dropna()
    params = {
        "modalidad": C.MODALIDAD_OBJETIVO,
        "objetivo": "ET",
        "label_et": C.LABEL_ET,
        "max_casos": args.max_casos,
        "n_casos_procesados": int(len(df)),
        "casos_con_et": int(len(con_et)),
        "otsu_t1c_mediana": round(float(df["otsu_t1c_bajo"].dropna().median()), 1) if df["otsu_t1c_bajo"].notna().any() else None,
        "otsu_t1c_alto_mediana": round(float(otsu_alto.median()), 1) if len(otsu_alto) else None,
        "otsu_t1c_rango": [round(float(otsu_alto.min()), 1), round(float(otsu_alto.max()), 1)] if len(otsu_alto) else None,
        "separabilidad_fdr_mediana": {r["modalidad"]: r["fdr_mediana"] for r in sep_rows},
        "mejor_modalidad_et": mejor_mod,
        "et_volumen_mm3": {
            "mediana": round(float(con_et["et_volumen_mm3"].median()), 1) if len(con_et) else None,
            "p25": round(float(con_et["et_volumen_mm3"].quantile(0.25)), 1) if len(con_et) else None,
            "p75": round(float(con_et["et_volumen_mm3"].quantile(0.75)), 1) if len(con_et) else None,
            "min": round(float(con_et["et_volumen_mm3"].min()), 1) if len(con_et) else None,
            "max": round(float(con_et["et_volumen_mm3"].max()), 1) if len(con_et) else None,
        },
        "casos_demostrativos": list(dict.fromkeys(elegidos.values())),
        "casos_demostrativos_detalle": elegidos,
    }
    with open(dirs["outputs"] / "parametros_eda.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # Figuras.
    print("\n[EDA] generando figuras...")
    hist_final = {m: (np.concatenate(v) if v else np.array([])) for m, v in hist_acum.items()}
    figs = {
        "histogramas": fig_histogramas(hist_final, dirs["figuras"]),
        "separabilidad": fig_separabilidad(sep_df, dirs["figuras"]),
        "volumen": fig_volumen_et(df, dirs["figuras"]),
        "otsu": fig_otsu_t1c(df, dirs["figuras"]),
        "demostrativos": fig_casos_demostrativos(elegidos, dirs["figuras"]),
    }
    # Limpiar temporales de los casos demostrativos usados en figuras.
    for cid in params["casos_demostrativos"]:
        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    print("[EDA] generando reporte HTML...")
    construir_reporte(params, df, sep_df, figs, dirs["reporte"])

    print("\n========== RESUMEN EDA ==========")
    print(f"  casos procesados : {params['n_casos_procesados']}")
    print(f"  casos con ET     : {params['casos_con_et']}")
    print(f"  Otsu T1c (alto)  : mediana={params['otsu_t1c_alto_mediana']}  rango={params['otsu_t1c_rango']}")
    print(f"  mejor modalidad  : {params['mejor_modalidad_et']}  (FDR={params['separabilidad_fdr_mediana']})")
    print(f"  demostrativos    : {params['casos_demostrativos_detalle']}")
    print(f"  salidas en       : {dirs['base']}")
    print("=================================")


if __name__ == "__main__":
    main()
