"""
run_all.py  —  Pipeline BraTS ET
=================================
Limpieza -> Segmentacion ET (métodos clásicos + semi-automático) -> Visualización

USO BÁSICO:
    python run_all.py                          # todo automático
    python run_all.py --skip-clean             # salta limpieza (ya hecha)

USO CON SEMILLA (mejor Dice):
    1. Abre viewer/viewer.html en el navegador
    2. Carga los archivos .nii.gz del caso
    3. Activa "Mapa dif" en Modalidad
    4. Haz clic en el punto más brillante del tumor
    5. Copia el comando que aparece y ejecútalo:

    python run_all.py --skip-clean --seed-case BraTS-GLI-02119-101 --seed-z 114 --seed-y 88 --seed-x 42

VARIABLES DE ENTORNO (si las rutas difieren):
    $env:BRATS_PROJECT_ROOT = "C:/ruta/al/proyecto"
    $env:BRATS_DATASET_DIR  = "C:/ruta/al/proyecto/images"
"""
from __future__ import annotations
import argparse, os, sys, time, glob as _glob

# Windows: la consola cp1252 no puede imprimir '✓'/'→' cuando stdout se redirige
# a un archivo. Forzar UTF-8 evita UnicodeEncodeError sin depender de PYTHONUTF8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from brats_pipeline import config, io_utils, clean_pipeline
from brats_pipeline.seg_et_pipeline import correr_pipeline_et
from brats_pipeline.viz_mosaics import (
    figura_mapa_diferencia,
    figura_comparativa_metodos,
    figura_metricas_resumen,
    overlay_3vistas,
)
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")


# ── CLI ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Pipeline BraTS ET")
    p.add_argument("--skip-clean",   action="store_true",
                   help="Saltar limpieza (usar volúmenes ya limpios)")
    p.add_argument("--skip-seg",     action="store_true")
    p.add_argument("--skip-viz",     action="store_true")
    p.add_argument("--pct",          type=float, default=90.0,
                   help="Percentil umbral sustracción (default 90)")
    p.add_argument("--sigma",        type=float, default=0.5)
    # Semilla manual para region growing (del visor)
    p.add_argument("--seed-case",    type=str, default=None,
                   help="case_id para el que aplicar la semilla")
    p.add_argument("--seed-z",       type=int, default=None)
    p.add_argument("--seed-y",       type=int, default=None)
    p.add_argument("--seed-x",       type=int, default=None)
    # Reconstrucción de superficie de Poisson (GT vs pred) para Dice alto
    p.add_argument("--poisson-thr",  type=float, default=0.75,
                   help="Umbral de Dice para generar la reconstrucción de "
                        "superficie de Poisson (default 0.75)")
    p.add_argument("--skip-poisson", action="store_true")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────
def cargar_limpio(case_id, mod):
    path = os.path.join(config.OUT_LIMPIEZA, case_id, f"{case_id}-{mod}.nii.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No encontré {path}.\n"
            f"  → Corre sin --skip-clean para generar los volúmenes limpios.")
    return sitk.ReadImage(path, sitk.sitkFloat32)


def cargar_raw(case_id, mod):
    """Imagen cruda SIN normalizar (para sustracción con norm. conjunta)."""
    hits = _glob.glob(os.path.join(config.DATASET_DIR, case_id, f"*-{mod}.nii*"))
    if not hits:
        return None
    return sitk.ReadImage(hits[0], sitk.sitkFloat32)


def cargar_seg_gt(case_id, ref_img=None):
    for base in (config.OUT_LIMPIEZA, config.DATASET_DIR):
        hits = _glob.glob(os.path.join(base, case_id, "*-seg.nii*"))
        if hits:
            seg = sitk.ReadImage(hits[0])
            if ref_img is not None:
                seg = sitk.Resample(
                    sitk.Cast(seg, sitk.sitkInt16), ref_img,
                    sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
            return seg
    raise FileNotFoundError(f"No encontré seg GT para {case_id}")


# ── PASO 1: Limpieza ───────────────────────────────────────────────
def paso_limpieza(casos):
    print("\n" + "="*60)
    print("PASO 1  LIMPIEZA  (Wiener + N4 + normalización percentil)")
    print("="*60)
    for case_id in casos:
        print(f"\n  [{case_id}]")
        clean_pipeline.limpiar_caso(
            case_id,
            base_dir=config.DATASET_DIR,
            out_root=config.OUT_LIMPIEZA,
            esquema_norm="percentil",
            verbose=True,
        )
    print(f"\n  ✓ {len(casos)} caso(s) limpios en {config.OUT_LIMPIEZA}")


# ── PASO 2: Segmentación ───────────────────────────────────────────
def paso_segmentacion(casos, args):
    print("\n" + "="*60)
    print("PASO 2  SEGMENTACIÓN ET")
    print("="*60)

    # Construir mapa de semillas si se pasaron por CLI
    semillas = {}
    if args.seed_case and args.seed_z is not None:
        semillas[args.seed_case] = (args.seed_z, args.seed_y, args.seed_x)
        print(f"  Semilla manual para {args.seed_case}: "
              f"z={args.seed_z}, y={args.seed_y}, x={args.seed_x}")

    todos_df, todos_mascaras = [], {}

    for case_id in casos:
        try:
            t1c = cargar_limpio(case_id, "t1c")
            t1n = cargar_limpio(case_id, "t1n")
            seg = cargar_seg_gt(case_id)
        except FileNotFoundError as e:
            print(f"  ✗ {e}"); continue

        t1c_raw = cargar_raw(case_id, "t1c")
        t1n_raw = cargar_raw(case_id, "t1n")
        try:
            t2f = cargar_limpio(case_id, "t2f")
        except Exception:
            t2f = None

        semilla = semillas.get(case_id)

        t0 = time.time()
        mascaras, gt_et, df = correr_pipeline_et(
            t1c, t1n, seg,
            t1c_raw=t1c_raw,
            t1n_raw=t1n_raw,
            t2f=t2f,
            semilla_zyx=semilla,
            case_id=case_id,
            auto_pct=args.pct,
            sigma=args.sigma,
            verbose=True,
        )
        df["tiempo_s"] = round(time.time() - t0, 1)
        todos_df.append(df)
        todos_mascaras[case_id] = mascaras

        # Guardar máscaras como NIfTI
        dst = os.path.join(config.OUT_SEG, case_id)
        os.makedirs(dst, exist_ok=True)
        for nombre, pred in mascaras.items():
            if nombre.startswith("_"): continue
            ps = io_utils.desde_numpy(pred.astype(np.uint8), ref=t1c)
            sitk.WriteImage(sitk.Cast(ps, sitk.sitkUInt8),
                            os.path.join(dst, f"{case_id}-et_{nombre}.nii.gz"),
                            useCompression=True)

    if todos_df:
        df_all = pd.concat(todos_df, ignore_index=True)
        os.makedirs(config.OUT_TABLAS, exist_ok=True)
        csv_path = os.path.join(config.OUT_TABLAS, "metricas_ET.csv")
        df_all.to_csv(csv_path, index=False)

        print("\n" + "-"*60)
        print("RESUMEN  Dice-ET")
        print("-"*60)
        pivot = df_all.pivot_table(
            index="case_id", columns="metodo", values="dice_ET")
        print(pivot.round(3).to_string())
        print("\nPromedios:")
        print(df_all.groupby("metodo")["dice_ET"].mean().round(3).to_string())
        print(f"\nCSV: {csv_path}")

    return todos_mascaras


# ── PASO 3: Visualización ──────────────────────────────────────────
def paso_visualizacion(casos, todos_mascaras):
    print("\n" + "="*60)
    print("PASO 3  VISUALIZACIÓN")
    print("="*60)
    os.makedirs(config.OUT_FIG, exist_ok=True)

    for case_id in casos:
        if case_id not in todos_mascaras: continue
        mascaras = todos_mascaras[case_id]
        arr_t1c  = mascaras["_arr_t1c"]
        mapa_dif = mascaras["_mapa_dif"]
        gt_et    = mascaras["_gt_et"]
        print(f"\n  [{case_id}]")

        try:
            arr_t1n = io_utils.a_numpy(
                cargar_raw(case_id, "t1n") or cargar_limpio(case_id, "t1n")
            ).astype("float32")
        except Exception:
            arr_t1n = np.zeros_like(arr_t1c)

        # Umbral efectivo del mapa (recalculado para la figura)
        cerebro = arr_t1c > 0
        vals_pos = mapa_dif[cerebro & (mapa_dif > 0)]
        umbral_fig = float(np.percentile(vals_pos, 90)) if len(vals_pos) > 0 else 0.0

        # Mejor predicción disponible (semilla > gmm_2d > gmm > otsu)
        def _get_pred(key):
            v = mascaras.get(key)
            return v if v is not None else None
        pred_best = (
            _get_pred("semilla") if _get_pred("semilla") is not None else
            _get_pred("gmm_T1c") if _get_pred("gmm_T1c") is not None else
            _get_pred("otsu_T1c") if _get_pred("otsu_T1c") is not None else
            np.zeros_like(gt_et)
        )

        # Figura 1: mapa diferencia
        figura_mapa_diferencia(
            arr_t1n, arr_t1c, mapa_dif, gt_et, pred_best,
            umbral=umbral_fig, case_id=case_id,
            path_out=os.path.join(config.OUT_FIG, f"{case_id}_mapa_dif.png"),
        )
        print(f"    ✓ mapa diferencia")

        # Figura 2: comparativa todos los métodos
        figura_comparativa_metodos(
            arr_t1c, gt_et, mascaras,
            case_id=case_id,
            path_out=os.path.join(config.OUT_FIG, f"{case_id}_comparativa.png"),
        )
        print(f"    ✓ comparativa métodos")

        # Figura 3: 3 vistas del mejor método
        try:
            t1c_ref = cargar_limpio(case_id, "t1c")
            seg_full = io_utils.a_numpy(
                cargar_seg_gt(case_id, ref_img=t1c_ref)).astype("int16")
        except Exception:
            seg_full = (gt_et * 3).astype("int16")

        metodo_nombre = ("semilla" if "semilla" in mascaras else
                         "gmm_2d" if "gmm_2d" in mascaras else "gmm_T1c")
        overlay_3vistas(
            arr_t1c, seg_full, pred=pred_best,
            titulo=f"{case_id}  —  3 vistas  |  {metodo_nombre}",
            path_out=os.path.join(config.OUT_FIG, f"{case_id}_3vistas.png"),
        )
        print(f"    ✓ 3 vistas ({metodo_nombre})")

    # Figura resumen Dice
    csv_path = os.path.join(config.OUT_TABLAS, "metricas_ET.csv")
    if os.path.exists(csv_path):
        figura_metricas_resumen(
            pd.read_csv(csv_path),
            path_out=os.path.join(config.OUT_FIG, "resumen_dice.png"),
        )
        print(f"\n  ✓ resumen Dice")

    print(f"\n  ✓ Figuras en {config.OUT_FIG}")


# ── PASO 4: Reconstrucción de superficie de Poisson ────────────────
def paso_poisson(casos, todos_mascaras, threshold=0.75):
    """
    Para cada (caso, método) cuyo Dice-ET supere `threshold`, reconstruye la
    superficie de Poisson del tumor predicho y del ground-truth y guarda una
    figura comparativa 3D + mallas .ply.
    """
    from brats_pipeline import viz_poisson
    from brats_pipeline.seg_metrics import dice as _dice

    print("\n" + "="*60)
    print(f"PASO 4  RECONSTRUCCIÓN DE SUPERFICIE DE POISSON  (Dice > {threshold})")
    print("="*60)
    out_dir = os.path.join(config.OUT_FIG, "poisson")
    os.makedirs(out_dir, exist_ok=True)

    generadas = 0
    for case_id in casos:
        if case_id not in todos_mascaras:
            continue
        mascaras = todos_mascaras[case_id]
        gt_et = mascaras.get("_gt_et")
        if gt_et is None:
            continue
        for nombre, pred in mascaras.items():
            if nombre.startswith("_"):
                continue
            d = _dice(pred, gt_et)
            if d <= threshold:
                continue
            path_out = os.path.join(out_dir, f"{case_id}_{nombre}_poisson.png")
            usado_o3d = viz_poisson.comparar_superficies(
                gt_et, pred, case_id=case_id, metodo=nombre,
                dice_val=float(d), path_out=path_out)
            motor = "open3d" if usado_o3d else "matplotlib"
            print(f"  ✓ {case_id:24s} {nombre:18s} Dice={d:.3f}  [{motor}]")
            generadas += 1

    if generadas == 0:
        print(f"  (ningún método superó Dice {threshold}; no se generaron "
              f"superficies de Poisson)")
    else:
        print(f"\n  ✓ {generadas} superficie(s) en {out_dir}")


# ── MAIN ───────────────────────────────────────────────────────────
def main():
    args = parse_args()
    config.asegurar_dirs()

    casos = config.detectar_casos()
    if not casos:
        print(f"\n✗  No hay casos en: {config.DATASET_DIR}")
        print("  Estructura esperada:")
        print("  images/BraTS-GLI-XXXXX-XXX/")
        print("    *-t1n.nii.gz  *-t1c.nii.gz  *-t2w.nii.gz  *-t2f.nii.gz  *-seg.nii.gz")
        sys.exit(1)

    print(f"\nCasos detectados en images/ ({len(casos)}):")
    for c in casos:
        print(f"  • {c}")

    mascaras = {}
    if not args.skip_clean:  paso_limpieza(casos)
    else: print("\n[skip] Limpieza.")

    if not args.skip_seg:    mascaras = paso_segmentacion(casos, args)
    else: print("\n[skip] Segmentación.")

    if not args.skip_viz and mascaras:
        paso_visualizacion(casos, mascaras)
    else: print("\n[skip] Visualización.")

    if not args.skip_poisson and mascaras:
        paso_poisson(casos, mascaras, threshold=args.poisson_thr)
    else: print("\n[skip] Poisson.")

    print("\n" + "="*60)
    print("LISTO")
    print(f"  Figuras  → {config.OUT_FIG}")
    print(f"  Métricas → {config.OUT_TABLAS}/metricas_ET.csv")
    print(f"  Máscaras → {config.OUT_SEG}")
    print("="*60)


if __name__ == "__main__":
    main()
