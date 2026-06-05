"""
fast_marching_et.py
===================
Segmentación del Enhancing Tumor (ET) por FastMarching sobre el mapa T1c-T1n.

USO:
    # Con semilla automática (corre en todos los casos sin intervención)
    python fast_marching_et.py --images images/

    # Con semilla aleatoria desde la máscara GT (para evaluación)
    python fast_marching_et.py --images images/ --seed-mode random

    # Con semilla manual (coordenadas conocidas)
    python fast_marching_et.py --images images/ --seed-case BraTS-GLI-02119-101 --seed-z 114 --seed-y 93 --seed-x 29

    # Buscar umbral óptimo para un caso (necesita GT)
    python fast_marching_et.py --images images/ --optimize-threshold

MODOS DE SEMILLA:
    auto    : score = T1c_limpio × mapa_dif, excluye vasos (mapa>0.55)
    random  : punto aleatorio dentro del GT-ET (para evaluación masiva)
    manual  : coordenadas (z,y,x) pasadas por CLI

SALIDAS:
    output/fast_marching/
        <caso>_ET_pred.nii.gz     — máscara ET predicha
        <caso>_mapa_dif.npy       — mapa T1c-T1n para visualización
    metricas_FM.csv               — Dice, Jaccard, volumen por caso
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter
import SimpleITK as sitk


# ================================================================== #
# Normalización conjunta                                              #
# ================================================================== #

def normalizar_conjunto(a1, a2, mask, p=99.5):
    vmax = max(float(np.percentile(a1[mask], p)),
               float(np.percentile(a2[mask], p)))
    if vmax <= 0: vmax = 1.0
    return (np.clip(a1/vmax, 0, 1).astype(np.float32),
            np.clip(a2/vmax, 0, 1).astype(np.float32))


def mapa_diferencia(t1c_raw, t1n_raw, sigma=0.8):
    cerebro = t1c_raw > 0
    t1c_n, t1n_n = normalizar_conjunto(t1c_raw, t1n_raw, cerebro)
    mapa = t1c_n - t1n_n
    if sigma > 0:
        mapa = gaussian_filter(mapa, sigma=sigma)
    return mapa


# ================================================================== #
# Semillas                                                            #
# ================================================================== #

def semilla_automatica(t1c_limpio, t1c_raw, t1n_raw):
    """
    Score = T1c_limpio × mapa_dif, excluyendo vasos (mapa > 0.55).
    Suavizado espacial para robustez.
    """
    cerebro = t1c_limpio > 0
    mapa = mapa_diferencia(t1c_raw, t1n_raw, sigma=0.5)
    mascara = cerebro & (mapa > 0.05) & (mapa < 0.55)
    score = t1c_limpio * mapa
    score_suav = uniform_filter(
        np.where(mascara, score, 0).astype(np.float32), size=5)
    score_suav = np.where(mascara, score_suav, -np.inf)
    idx = np.unravel_index(np.argmax(score_suav), score_suav.shape)
    return idx  # (z, y, x)


def semilla_aleatoria_en_gt(seg_arr, seed=None):
    """
    Punto aleatorio dentro del GT-ET (etiqueta 3).
    Útil para evaluación masiva sin intervención manual.
    """
    gt_et = (np.round(seg_arr) == 3)
    if gt_et.sum() == 0:
        return None
    coords = np.array(np.where(gt_et)).T  # (N, 3)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(coords))
    z, y, x = coords[idx]
    return (int(z), int(y), int(x))


# ================================================================== #
# Semilla inteligente — centroide del componente más denso           #
# ================================================================== #

def semilla_centroide_gt(seg_arr):
    """
    Semilla en el centroide exacto del GT-ET.

    Simula la mejor semilla manual posible — lo que haría un radiólogo
    que coloca la semilla en el centro del tumor visualmente.
    Es determinística (siempre da el mismo punto) y garantiza
    estar dentro del tumor.

    Retorna (z, y, x) en orden numpy, o None si GT-ET está vacío.
    """
    from scipy.ndimage import center_of_mass
    gt_et = (np.round(seg_arr) == 3)
    if gt_et.sum() == 0:
        return None
    cm = center_of_mass(gt_et)
    z0, y0, x0 = int(round(cm[0])), int(round(cm[1])), int(round(cm[2]))
    # Si el centroide no cae dentro del GT (tumor en forma de C, etc.)
    # tomar el voxel más cercano al centroide que sí esté en el GT
    if not gt_et[z0, y0, x0]:
        coords = np.array(np.where(gt_et)).T
        dists = np.linalg.norm(coords - np.array([z0, y0, x0]), axis=1)
        closest = coords[np.argmin(dists)]
        z0, y0, x0 = int(closest[0]), int(closest[1]), int(closest[2])
    return (z0, y0, x0)


# ================================================================== #
# FastMarching ET                                                     #
# ================================================================== #

def fast_marching_et(t1c_raw, t1n_raw, semilla_zyx,
                      tiempo_umbral=35.0, sigma=0.8):
    """
    Segmenta ET usando FastMarching con velocidad = mapa T1c-T1n.

    El frente avanza rápido en ET (mapa alto ~0.24) y lento en tejido
    sano (mapa bajo ~0.04). Umbralizando el tiempo de llegada obtenemos
    el ET sin depender de conectividad de intensidad.
    """
    cerebro = t1c_raw > 0
    mapa = mapa_diferencia(t1c_raw, t1n_raw, sigma=sigma)
    mapa_pos = np.clip(mapa, 0.001, None).astype(np.float32)

    speed_sitk = sitk.GetImageFromArray(mapa_pos)
    z0, y0, x0 = semilla_zyx
    fm = sitk.FastMarchingImageFilter()
    fm.AddTrialPoint((int(x0), int(y0), int(z0), 0))  # ITK: x,y,z
    fm.SetStoppingValue(tiempo_umbral * 3.0)
    tiempo_arr = sitk.GetArrayFromImage(fm.Execute(speed_sitk))

    pred = (tiempo_arr <= tiempo_umbral) & cerebro
    return pred.astype(np.uint8), mapa, tiempo_arr


def buscar_umbral_optimo(t1c_raw, t1n_raw, seg_arr, semilla_zyx,
                          grid=None):
    """Busca el tiempo_umbral con mayor Dice-ET."""
    if grid is None:
        grid = [10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]
    cerebro = t1c_raw > 0
    mapa = mapa_diferencia(t1c_raw, t1n_raw, sigma=0.8)
    mapa_pos = np.clip(mapa, 0.001, None).astype(np.float32)
    gt_et = (np.round(seg_arr) == 3).astype(np.uint8)

    speed_sitk = sitk.GetImageFromArray(mapa_pos)
    z0, y0, x0 = semilla_zyx
    fm = sitk.FastMarchingImageFilter()
    fm.AddTrialPoint((int(x0), int(y0), int(z0), 0))
    fm.SetStoppingValue(max(grid) * 2.0)
    tiempo_arr = sitk.GetArrayFromImage(fm.Execute(speed_sitk))

    def dice(a, b):
        s = a.sum() + b.sum()
        return 2 * np.logical_and(a, b).sum() / s if s > 0 else 0.0

    resultados = {}
    for t in grid:
        pred = (tiempo_arr <= t) & cerebro
        resultados[t] = round(dice(pred.astype(np.uint8), gt_et), 4)

    mejor_t = max(resultados, key=resultados.get)
    return mejor_t, resultados[mejor_t], resultados


# ================================================================== #
# Helpers de I/O                                                      #
# ================================================================== #

def find_file(case_dir, suffix):
    hits = glob.glob(os.path.join(case_dir, f"*-{suffix}.nii*"))
    return hits[0] if hits else None


def load_raw(path):
    return sitk.GetArrayFromImage(
        sitk.ReadImage(path, sitk.sitkFloat32))


def load_limpio(case_id, out_limpieza, mod):
    """Carga versión limpia si existe, si no la cruda."""
    p = os.path.join(out_limpieza, case_id, f"{case_id}-{mod}.nii.gz")
    if os.path.exists(p):
        return sitk.GetArrayFromImage(sitk.ReadImage(p, sitk.sitkFloat32))
    return None


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 2 * int(np.logical_and(a, b).sum()) / s if s > 0 else 0.0


def jaccard(a, b):
    u = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / u if u > 0 else 0.0


# ================================================================== #
# Pipeline por caso                                                   #
# ================================================================== #

def procesar_caso(case_id, case_dir, args, out_dir, limpieza_dir):
    print(f"\n  [{case_id}]")

    # Cargar imágenes
    t1c_path = find_file(case_dir, "t1c")
    t1n_path = find_file(case_dir, "t1n")
    seg_path = find_file(case_dir, "seg")

    if not t1c_path or not t1n_path:
        print(f"  ✗ Faltan t1c o t1n en {case_dir}")
        return None

    t1c_raw = load_raw(t1c_path)
    t1n_raw = load_raw(t1n_path)

    # T1c limpio (para score de semilla automática)
    t1c_limpio = load_limpio(case_id, limpieza_dir, "t1c")
    if t1c_limpio is None:
        # Normalizar por percentil si no hay versión limpia
        cerebro = t1c_raw > 0
        p1 = float(np.percentile(t1c_raw[cerebro], 1))
        p99 = float(np.percentile(t1c_raw[cerebro], 99))
        t1c_limpio = np.clip((t1c_raw - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)

    # GT si existe
    seg_arr = None
    gt_et = None
    if seg_path:
        seg_arr = np.round(load_raw(seg_path)).astype(np.int16)
        gt_et = (seg_arr == 3).astype(np.uint8)
        print(f"  GT-ET: {gt_et.sum()} vóxeles")

    # ── Determinar semilla ─────────────────────────────────────────
    semilla = None
    modo = args.seed_mode

    if args.seed_case == case_id and args.seed_z is not None:
        # Semilla manual explícita
        semilla = (args.seed_z, args.seed_y, args.seed_x)
        modo = "manual"

    elif modo == "centroide":
        if seg_arr is not None:
            semilla = semilla_centroide_gt(seg_arr)
        if semilla is None:
            semilla = semilla_automatica(t1c_limpio, t1c_raw, t1n_raw)

    elif modo == "random":
        if seg_arr is not None:
            semilla = semilla_aleatoria_en_gt(seg_arr, seed=args.random_seed)
            if semilla is None:
                print(f"  GT-ET vacío, usando semilla automática")
                modo = "auto"
        else:
            print(f"  Sin GT para semilla aleatoria, usando auto")
            modo = "auto"

    if semilla is None:
        semilla = semilla_automatica(t1c_limpio, t1c_raw, t1n_raw)
        modo = "auto"

    z0, y0, x0 = semilla
    print(f"  Semilla [{modo}]: z={z0} y={y0} x={x0}")

    # ── Optimizar umbral si se pide ────────────────────────────────
    umbral = args.threshold
    if args.optimize_threshold and gt_et is not None and gt_et.sum() > 0:
        best_t, best_d, grid_res = buscar_umbral_optimo(
            t1c_raw, t1n_raw, seg_arr, semilla)
        print(f"  Umbral óptimo: t={best_t}  Dice={best_d:.3f}")
        print(f"  Grid: {grid_res}")
        umbral = best_t

    # ── FastMarching ───────────────────────────────────────────────
    t0 = time.time()
    pred, mapa, tiempo_arr = fast_marching_et(
        t1c_raw, t1n_raw, semilla, tiempo_umbral=umbral)
    elapsed = time.time() - t0
    print(f"  FM pred: {pred.sum()} vox  umbral={umbral}  ({elapsed:.1f}s)")

    # ── Métricas ───────────────────────────────────────────────────
    d, j = 0.0, 0.0
    if gt_et is not None:
        d = dice(pred, gt_et)
        j = jaccard(pred, gt_et)
        print(f"  Dice-ET={d:.3f}  Jaccard={j:.3f}")

    # ── Guardar ────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    pred_sitk = sitk.GetImageFromArray(pred)
    # Copiar geometría de T1c original
    t1c_img = sitk.ReadImage(t1c_path, sitk.sitkFloat32)
    pred_sitk.CopyInformation(t1c_img)
    sitk.WriteImage(sitk.Cast(pred_sitk, sitk.sitkUInt8),
                    os.path.join(out_dir, f"{case_id}_ET_pred.nii.gz"),
                    useCompression=True)
    np.save(os.path.join(out_dir, f"{case_id}_mapa_dif.npy"), mapa)

    return {
        "case_id":       case_id,
        "seed_mode":     modo,
        "seed_z":        z0, "seed_y": y0, "seed_x": x0,
        "threshold":     umbral,
        "pred_vox":      int(pred.sum()),
        "gt_vox":        int(gt_et.sum()) if gt_et is not None else -1,
        "dice_ET":       round(d, 4),
        "jaccard_ET":    round(j, 4),
        "tiempo_s":      round(elapsed, 1),
    }


# ================================================================== #
# Main                                                                #
# ================================================================== #

def parse_args():
    p = argparse.ArgumentParser(description="FastMarching ET — BraTS 2024")
    p.add_argument("--images",     default="images",
                   help="Carpeta con los casos (default: images/)")
    p.add_argument("--output",     default="output/fast_marching",
                   help="Carpeta de salida (default: output/fast_marching)")
    p.add_argument("--limpieza",   default="output/limpieza",
                   help="Carpeta con volúmenes limpios (opcional)")
    p.add_argument("--threshold",  type=float, default=35.0,
                   help="Umbral de tiempo FastMarching (default: 35)")
    p.add_argument("--seed-mode",  choices=["auto", "random"], default="auto",
                   help="Modo semilla: auto (score T1c*mapa) o random (GT aleatorio)")
    p.add_argument("--random-seed", type=int, default=42,
                   help="Semilla aleatoria para reproducibilidad (default: 42)")
    p.add_argument("--optimize-threshold", action="store_true",
                   help="Buscar umbral óptimo por caso (necesita GT)")
    # Semilla manual para un caso específico
    p.add_argument("--seed-case",  type=str, default=None)
    p.add_argument("--seed-z",     type=int, default=None)
    p.add_argument("--seed-y",     type=int, default=None)
    p.add_argument("--seed-x",     type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Detectar casos
    casos = []
    for nombre in sorted(os.listdir(args.images)):
        ruta = os.path.join(args.images, nombre)
        if os.path.isdir(ruta):
            niis = glob.glob(os.path.join(ruta, "*.nii*"))
            if niis:
                casos.append((nombre, ruta))

    if not casos:
        print(f"✗ No hay casos en: {args.images}")
        sys.exit(1)

    print(f"\nCasos detectados: {len(casos)}")
    for n, _ in casos: print(f"  • {n}")

    # Siempre corre los 3 modos para comparar
    modos = ["auto", "centroide", "random"]
    todos = []

    for modo in modos:
        print(f"\n{'='*60}")
        print(f"MODO: {modo.upper()}")
        print(f"{'='*60}")
        args.seed_mode = modo
        out_modo = os.path.join(args.output, modo)
        for case_id, case_dir in casos:
            r = procesar_caso(case_id, case_dir, args, out_modo, args.limpieza)
            if r:
                todos.append(r)

    if todos:
        df = pd.DataFrame(todos)
        os.makedirs(args.output, exist_ok=True)
        csv_path = os.path.join(args.output, "metricas_FM_comparacion.csv")
        df.to_csv(csv_path, index=False)

        print("\n" + "="*60)
        print("COMPARACION AUTO vs CENTROIDE vs RANDOM")
        print("="*60)
        pivot = df.pivot_table(
            index="case_id", columns="seed_mode",
            values="dice_ET", aggfunc="mean")
        print(pivot.round(3).to_string())
        print(f"\nPromedios:")
        print(df.groupby("seed_mode")["dice_ET"].mean().round(3).to_string())
        print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
