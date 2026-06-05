"""
run_all.py  —  Pipeline FastMarching ET
========================================
Limpieza → FastMarching (auto + random) → Visualización

USO:
    # Primera vez (con limpieza)
    python run_all.py

    # Sin relimpiar
    python run_all.py --skip-clean

    # Con semilla manual para un caso
    python run_all.py --skip-clean --seed-case BraTS-GLI-02119-101 --seed-z 114 --seed-y 93 --seed-x 29

    # Solo semilla automática (sin random)
    python run_all.py --skip-clean --no-random

VARIABLES DE ENTORNO (opcional):
    $env:BRATS_PROJECT_ROOT = "C:/ruta/proyecto"
    $env:BRATS_DATASET_DIR  = "C:/ruta/proyecto/images"
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from brats_pipeline import config, io_utils, clean_pipeline
from brats_pipeline.seg_metrics import dice, jaccard

from fast_marching_et import (
    mapa_diferencia, semilla_automatica, semilla_centroide_gt,
    semilla_aleatoria_en_gt, fast_marching_et,
    buscar_umbral_optimo
)


# ── CLI ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-clean",    action="store_true")
    p.add_argument("--skip-viz",      action="store_true")
    p.add_argument("--no-random",     action="store_true",
                   help="Solo semilla automática, no correr random")
    p.add_argument("--threshold",     type=float, default=35.0)
    p.add_argument("--optimize-threshold", action="store_true")
    p.add_argument("--seed-case",     type=str, default=None)
    p.add_argument("--seed-z",        type=int, default=None)
    p.add_argument("--seed-y",        type=int, default=None)
    p.add_argument("--seed-x",        type=int, default=None)
    p.add_argument("--random-seed",   type=int, default=42)
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────
def detectar_casos():
    dataset_dir = os.environ.get("BRATS_DATASET_DIR",
                                  os.path.join(ROOT, "images"))
    casos = []
    if not os.path.exists(dataset_dir):
        return casos
    for nombre in sorted(os.listdir(dataset_dir)):
        ruta = os.path.join(dataset_dir, nombre)
        if os.path.isdir(ruta) and glob.glob(os.path.join(ruta, "*.nii*")):
            casos.append(nombre)
    return casos


def cargar_raw(case_id, mod):
    dataset_dir = os.environ.get("BRATS_DATASET_DIR",
                                  os.path.join(ROOT, "images"))
    hits = glob.glob(os.path.join(dataset_dir, case_id, f"*-{mod}.nii*"))
    if not hits: return None
    return sitk.GetArrayFromImage(sitk.ReadImage(hits[0], sitk.sitkFloat32))


def cargar_limpio(case_id, mod):
    out_limpieza = os.path.join(ROOT, "output", "limpieza")
    p = os.path.join(out_limpieza, case_id, f"{case_id}-{mod}.nii.gz")
    if os.path.exists(p):
        return sitk.GetArrayFromImage(sitk.ReadImage(p, sitk.sitkFloat32))
    return None


def cargar_seg(case_id):
    # Intentar limpio primero, luego raw
    for base in [os.path.join(ROOT, "output", "limpieza"),
                 os.environ.get("BRATS_DATASET_DIR", os.path.join(ROOT, "images"))]:
        hits = glob.glob(os.path.join(base, case_id, "*-seg.nii*"))
        if hits:
            return np.round(sitk.GetArrayFromImage(
                sitk.ReadImage(hits[0]))).astype(np.int16)
    return None


# ── PASO 1: Limpieza ───────────────────────────────────────────────
def paso_limpieza(casos):
    print("\n" + "="*60)
    print("PASO 1  LIMPIEZA  (Wiener + N4 + normalización percentil)")
    print("="*60)
    dataset_dir = os.environ.get("BRATS_DATASET_DIR",
                                  os.path.join(ROOT, "images"))
    out_limpieza = os.path.join(ROOT, "output", "limpieza")
    for case_id in casos:
        print(f"\n  [{case_id}]")
        clean_pipeline.limpiar_caso(
            case_id,
            base_dir=dataset_dir,
            out_root=out_limpieza,
            esquema_norm="percentil",
            verbose=True,
        )


# ── PASO 2: FastMarching ───────────────────────────────────────────
def paso_fast_marching(casos, args):
    print("\n" + "="*60)
    print("PASO 2  FAST MARCHING ET")
    print("="*60)

    out_fm = os.path.join(ROOT, "output", "fast_marching")
    os.makedirs(out_fm, exist_ok=True)
    dataset_dir = os.environ.get("BRATS_DATASET_DIR",
                                  os.path.join(ROOT, "images"))

    todos = []
    mascaras_por_caso = {}

    # Modos a correr
    modos = ["auto", "centroide"]
    if not args.no_random:
        modos.append("random")

    for case_id in casos:
        print(f"\n  [{case_id}]")
        t1c_raw = cargar_raw(case_id, "t1c")
        t1n_raw = cargar_raw(case_id, "t1n")
        if t1c_raw is None or t1n_raw is None:
            print(f"  ✗ Faltan t1c/t1n"); continue

        t1c_limpio = cargar_limpio(case_id, "t1c")
        if t1c_limpio is None:
            cerebro = t1c_raw > 0
            p1 = float(np.percentile(t1c_raw[cerebro], 1))
            p99 = float(np.percentile(t1c_raw[cerebro], 99))
            t1c_limpio = np.clip((t1c_raw-p1)/(p99-p1+1e-8), 0, 1).astype(np.float32)

        seg_arr = cargar_seg(case_id)
        gt_et = (seg_arr == 3).astype(np.uint8) if seg_arr is not None else None
        if gt_et is not None:
            print(f"  GT-ET: {gt_et.sum()} vóxeles")

        mascaras_caso = {"_t1c_raw": t1c_raw, "_t1n_raw": t1n_raw,
                         "_t1c_limpio": t1c_limpio, "_gt_et": gt_et}
        mapa = mapa_diferencia(t1c_raw, t1n_raw, sigma=0.8)
        mascaras_caso["_mapa"] = mapa

        for modo in modos:
            # Determinar semilla
            if args.seed_case == case_id and args.seed_z is not None:
                semilla = (args.seed_z, args.seed_y, args.seed_x)
                modo_str = "manual"
            elif modo == "centroide":
                semilla = semilla_centroide_gt(seg_arr) if seg_arr is not None else None
                if semilla is None:
                    semilla = semilla_automatica(t1c_limpio, t1c_raw, t1n_raw)
                modo_str = "centroide"
            elif modo == "random" and gt_et is not None and gt_et.sum() > 0:
                semilla = semilla_aleatoria_en_gt(seg_arr, seed=args.random_seed)
                modo_str = "random"
            else:
                semilla = semilla_automatica(t1c_limpio, t1c_raw, t1n_raw)
                modo_str = "auto"

            if semilla is None:
                continue

            z0, y0, x0 = semilla
            umbral = args.threshold

            if args.optimize_threshold and gt_et is not None and gt_et.sum() > 0:
                best_t, best_d, _ = buscar_umbral_optimo(
                    t1c_raw, t1n_raw, seg_arr, semilla)
                umbral = best_t

            t0 = time.time()
            pred, _, _ = fast_marching_et(t1c_raw, t1n_raw, semilla,
                                           tiempo_umbral=umbral)
            elapsed = time.time() - t0

            d = dice(pred, gt_et) if gt_et is not None else 0.0
            j = jaccard(pred, gt_et) if gt_et is not None else 0.0
            print(f"  [{modo_str}] z={z0},y={y0},x={x0}  "
                  f"pred={pred.sum():6d}  Dice={d:.3f}  ({elapsed:.1f}s)")

            # Guardar máscara
            hits = glob.glob(os.path.join(dataset_dir, case_id, "*-t1c.nii*"))
            if hits:
                ref = sitk.ReadImage(hits[0], sitk.sitkFloat32)
                pred_sitk = sitk.GetImageFromArray(pred)
                pred_sitk.CopyInformation(ref)
                sitk.WriteImage(sitk.Cast(pred_sitk, sitk.sitkUInt8),
                    os.path.join(out_fm, f"{case_id}_ET_{modo_str}.nii.gz"),
                    useCompression=True)

            mascaras_caso[f"pred_{modo_str}"] = pred
            todos.append({
                "case_id": case_id, "seed_mode": modo_str,
                "seed_z": z0, "seed_y": y0, "seed_x": x0,
                "threshold": umbral,
                "pred_vox": int(pred.sum()),
                "gt_vox": int(gt_et.sum()) if gt_et is not None else -1,
                "dice_ET": round(d, 4),
                "jaccard_ET": round(j, 4),
                "tiempo_s": round(elapsed, 1),
            })

        mascaras_por_caso[case_id] = mascaras_caso

    if todos:
        df = pd.DataFrame(todos)
        csv = os.path.join(out_fm, "metricas_FM.csv")
        df.to_csv(csv, index=False)
        print("\n" + "-"*60)
        print("RESUMEN Dice-ET")
        print("-"*60)
        if len(modos) > 1:
            pivot = df.pivot_table(index="case_id", columns="seed_mode",
                                    values="dice_ET", aggfunc="mean")
            print(pivot.round(3).to_string())
            print("\nPromedios:")
            print(df.groupby("seed_mode")["dice_ET"].mean().round(3).to_string())
        else:
            print(df[["case_id","dice_ET","pred_vox","gt_vox"]].to_string(index=False))
            print(f"\nPromedio Dice: {df['dice_ET'].mean():.3f}")
        print(f"\nCSV: {csv}")

    return mascaras_por_caso, pd.DataFrame(todos) if todos else pd.DataFrame()


# ── PASO 3: Visualización ──────────────────────────────────────────
def figura_caso(case_id, mascaras, out_fig, modos):
    t1c = mascaras.get("_t1c_limpio")
    t1n = mascaras.get("_t1n_raw")
    t1c_raw = mascaras.get("_t1c_raw")
    mapa = mascaras.get("_mapa")
    gt_et = mascaras.get("_gt_et")

    if t1c is None or mapa is None:
        return

    # Slice con más GT o con más pred
    if gt_et is not None and gt_et.sum() > 0:
        z_best = int(np.bincount(np.where(gt_et)[0]).argmax())
    else:
        z_best = t1c.shape[0] // 2

    def norm(arr, z):
        sl = arr[z]
        mn, mx = sl.min(), sl.max()
        return (sl - mn) / (mx - mn + 1e-8)

    n_preds = sum(1 for m in modos if f"pred_{m}" in mascaras)
    n_cols = 3 + n_preds  # T1n, T1c, mapa, pred×n
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    fig.suptitle(f"{case_id}  —  z={z_best}", fontsize=12)

    axes[0].imshow(norm(t1c if t1n is None else t1n, z_best), cmap="gray")
    axes[0].set_title("T1n"); axes[0].axis("off")

    axes[1].imshow(norm(t1c, z_best), cmap="gray")
    # overlay GT
    if gt_et is not None:
        gt_sl = gt_et[z_best].astype(float)
        axes[1].imshow(np.ma.masked_where(gt_sl == 0, gt_sl),
                       cmap="cool", alpha=0.4, vmin=0, vmax=1)
    axes[1].set_title("T1c + GT-ET (cian)"); axes[1].axis("off")

    axes[2].imshow(mapa[z_best], cmap="hot", vmin=0, vmax=0.6)
    axes[2].set_title(f"Mapa T1c−T1n"); axes[2].axis("off")

    colors = {"auto": "Reds", "centroide": "Oranges", "random": "Greens", "manual": "Blues"}
    for i, modo in enumerate([m for m in modos if f"pred_{m}" in mascaras]):
        ax = axes[3 + i]
        pred = mascaras[f"pred_{modo}"]
        bg = norm(t1c, z_best)
        ax.imshow(bg, cmap="gray")
        if gt_et is not None:
            gt_sl = gt_et[z_best].astype(float)
            ax.imshow(np.ma.masked_where(gt_sl == 0, gt_sl),
                      cmap="cool", alpha=0.35, vmin=0, vmax=1)
        pred_sl = pred[z_best].astype(float)
        ax.imshow(np.ma.masked_where(pred_sl == 0, pred_sl),
                  cmap=colors.get(modo, "Reds"), alpha=0.5, vmin=0, vmax=1)
        d = dice(pred, gt_et) if gt_et is not None else 0.0
        ax.set_title(f"FM [{modo}]\nDice={d:.3f}")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_fig, f"{case_id}_FM.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓ {os.path.basename(path)}")


def figura_resumen(df, out_fig):
    if df.empty: return
    fig, ax = plt.subplots(figsize=(max(8, len(df)*0.6), 5))
    modos = df["seed_mode"].unique()
    x = np.arange(df["case_id"].nunique())
    cases = df["case_id"].unique()
    w = 0.35
    colors = {"auto": "#2196F3", "centroide": "#FF9800", "random": "#4CAF50", "manual": "#FF5722"}

    for i, modo in enumerate(modos):
        sub = df[df["seed_mode"] == modo].groupby("case_id")["dice_ET"].mean()
        vals = [float(sub.loc[c]) if c in sub.index else 0.0 for c in cases]
        vals = [0.0 if (v != v) else v for v in vals]  # reemplaza NaN con 0
        offset = (i - len(modos)/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w * 0.9, label=modo,
                      color=colors.get(modo, "#9C27B0"), alpha=0.8)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("BraTS-GLI-", "") for c in cases],
                        rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Dice-ET")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("FastMarching ET — Comparación Auto vs Random")
    ax.axhline(df.groupby("seed_mode")["dice_ET"].mean().mean(),
               color="gray", linestyle="--", alpha=0.5, label="promedio")
    plt.tight_layout()
    path = os.path.join(out_fig, "resumen_FM.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ resumen_FM.png")


def paso_visualizacion(casos, mascaras_por_caso, df, args):
    print("\n" + "="*60)
    print("PASO 3  VISUALIZACIÓN")
    print("="*60)
    out_fig = os.path.join(ROOT, "output", "figuras")
    os.makedirs(out_fig, exist_ok=True)

    modos = ["auto", "centroide"]
    if not args.no_random: modos.append("random")
    if args.seed_case: modos.append("manual")

    for case_id in casos:
        if case_id not in mascaras_por_caso: continue
        print(f"\n  [{case_id}]")
        figura_caso(case_id, mascaras_por_caso[case_id], out_fig, modos)

    figura_resumen(df, out_fig)
    print(f"\n  ✓ Figuras en {out_fig}")


# ── MAIN ───────────────────────────────────────────────────────────
def main():
    args = parse_args()

    casos = detectar_casos()
    if not casos:
        dataset_dir = os.environ.get("BRATS_DATASET_DIR",
                                      os.path.join(ROOT, "images"))
        print(f"\n✗ No hay casos en: {dataset_dir}")
        sys.exit(1)

    print(f"\nCasos detectados ({len(casos)}):")
    for c in casos: print(f"  • {c}")

    if not args.skip_clean:
        paso_limpieza(casos)
    else:
        print("\n[skip] Limpieza.")

    mascaras, df = paso_fast_marching(casos, args)

    if not args.skip_viz and mascaras:
        paso_visualizacion(casos, mascaras, df, args)

    print("\n" + "="*60)
    print("LISTO")
    print(f"  Figuras   → output/figuras/")
    print(f"  Métricas  → output/fast_marching/metricas_FM.csv")
    print(f"  Máscaras  → output/fast_marching/")
    print("="*60)


if __name__ == "__main__":
    main()
