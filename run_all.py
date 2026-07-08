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
import argparse, datetime as _dt, glob as _glob, hashlib, json, os
import platform, subprocess, sys, time
from pathlib import Path

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
    p.add_argument("--config", default=os.path.join("configs", "pipeline.yaml"),
                   help="YAML de configuracion del pipeline")
    p.add_argument("--legacy", action="store_true",
                   help="Incluir los metodos legacy/cut definidos en methods_legacy")
    p.add_argument("--data-root", default=None,
                   help="Raiz con carpetas BraTS-GLI-* de entrada")
    p.add_argument("--output-root", default=None,
                   help="Raiz de salida; contiene limpieza/segmentacion/figuras/tablas")
    p.add_argument("--clean-root", default=None,
                   help="Raiz de volumenes limpios ya existentes cuando se usa --skip-clean")
    p.add_argument("--cohort", default=None,
                   help="CSV de cohorte/manifest con columna case_id")
    p.add_argument("--process-only", action="store_true",
                   help="Con --cohort, correr solo filas process=1")
    p.add_argument("--case-dir", default=None,
                   help="Carpeta explicita de un caso; su padre se usa como data-root")
    p.add_argument("--case-id", action="append", default=None,
                   help="Case_id explicito; se puede repetir")
    p.add_argument("--force", action="store_true",
                   help="Ignorar checkpoints .done y recomputar")
    p.add_argument("--skip-clean",   action="store_true",
                   help="Saltar limpieza (usar volúmenes ya limpios)")
    p.add_argument("--skip-seg",     action="store_true")
    p.add_argument("--skip-viz",     action="store_true")
    p.add_argument("--pct",          type=float, default=None,
                   help="Percentil umbral sustracción (default 90)")
    p.add_argument("--sigma",        type=float, default=None)
    # Semilla manual para region growing (del visor)
    p.add_argument("--seed-case",    type=str, default=None,
                   help="case_id para el que aplicar la semilla")
    p.add_argument("--seed-z",       type=int, default=None)
    p.add_argument("--seed-y",       type=int, default=None)
    p.add_argument("--seed-x",       type=int, default=None)
    # Reconstrucción de superficie de Poisson (GT vs pred) para Dice alto
    p.add_argument("--poisson-thr",  type=float, default=None,
                   help="Umbral de Dice para generar la reconstrucción de "
                        "superficie de Poisson (default 0.75)")
    p.add_argument("--skip-poisson", action="store_true")
    return p.parse_args()


def cargar_pipeline_config(path):
    """Carga el YAML minimo usado por este proyecto sin depender de PyYAML."""
    cfg = {}
    current = None
    if not path:
        return cfg
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"No encontre config: {full}")
    with open(full, encoding="utf-8") as f:
        for raw in f:
            line = raw.lstrip("\ufeff").split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" ") and line.endswith(":"):
                current = line[:-1].strip()
                cfg[current] = []
                continue
            if not line.startswith(" ") and ":" in line:
                key, value = line.split(":", 1)
                current = None
                cfg[key.strip()] = _parse_yaml_scalar(value.strip())
                continue
            stripped = line.strip()
            if current and stripped.startswith("- "):
                cfg[current].append(_parse_yaml_scalar(stripped[2:].strip()))
    return cfg


def _parse_yaml_scalar(value):
    if value == "":
        return ""
    low = value.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def metodos_desde_config(resolved_config, legacy=False):
    methods = list(resolved_config.get("methods", []))
    if legacy:
        methods.extend(resolved_config.get("methods_legacy", []))
    return methods or None


def resolver_config(args):
    cfg = cargar_pipeline_config(args.config)
    if args.pct is not None:
        cfg["auto_pct"] = float(args.pct)
    if args.sigma is not None:
        cfg["sigma"] = float(args.sigma)
    if args.poisson_thr is not None:
        cfg["poisson_threshold"] = float(args.poisson_thr)
        cfg["dice_3d_cutoff"] = float(args.poisson_thr)
    cfg["legacy"] = bool(args.legacy)
    cfg["methods_resolved"] = metodos_desde_config(cfg, legacy=args.legacy) or []
    return cfg


def aplicar_config_runtime(cfg):
    mapping = {
        "seed": ("SEED", int),
        "auto_pct": ("AUTO_PCT", float),
        "sigma": ("SIGMA", float),
        "poisson_threshold": ("POISSON_THRESHOLD", float),
        "enable_evidence_guard": ("ENABLE_EVIDENCE_GUARD", bool),
        "guard_min_lcc_fraction": ("GUARD_MIN_LCC_FRACTION", float),
        "guard_min_enhancement_ratio": ("GUARD_MIN_ENHANCEMENT_RATIO", float),
        "guard_max_volume_multiple": ("GUARD_MAX_VOLUME_MULTIPLE", float),
        "enable_best_iterate": ("ENABLE_BEST_ITERATE", bool),
        "best_iterate_w_lcc": ("BEST_ITERATE_W_LCC", float),
        "best_iterate_w_enhancement": ("BEST_ITERATE_W_ENHANCEMENT", float),
        "best_iterate_w_volume_stability": ("BEST_ITERATE_W_VOLUME_STABILITY", float),
        "best_iterate_patience": ("BEST_ITERATE_PATIENCE", int),
        "variational_spline_iters": ("VARIATIONAL_SPLINE_ITERS", int),
        "variational_spline_smoothing": ("VARIATIONAL_SPLINE_SMOOTHING", int),
        "bspline_chanvese_iters": ("BSPLINE_CHANVESE_ITERS", int),
        "bspline_chanvese_smoothing": ("BSPLINE_CHANVESE_SMOOTHING", int),
        "level_set_iters": ("LEVEL_SET_ITERS", int),
        "fast_marching_time_threshold": ("FAST_MARCHING_TIME_THRESHOLD", float),
        "n4_shrink": ("N4_SHRINK", int),
        "wiener_size": ("WIENER_SIZE", int),
        "normalization_scheme": ("NORM_SUSTRACCION", str),
    }
    for key, (attr, caster) in mapping.items():
        if key in cfg and cfg[key] is not None:
            setattr(config, attr, caster(cfg[key]))


def configurar_rutas(args, cfg):
    data_root = args.data_root or cfg.get("data_root") or os.path.join(ROOT, "images")
    if args.case_dir:
        case_dir = os.path.abspath(args.case_dir)
        data_root = os.path.dirname(case_dir)
    output_root = args.output_root or cfg.get("output_root") or os.path.join(ROOT, "output")
    clean_root = args.clean_root or cfg.get("clean_root") or os.path.join(output_root, "limpieza")

    config.DATASET_DIR = os.path.abspath(data_root)
    config.OUT_LIMPIEZA = os.path.abspath(clean_root)
    config.OUT_SEG = os.path.abspath(os.path.join(output_root, "segmentacion"))
    config.OUT_FIG = os.path.abspath(os.path.join(output_root, "figuras"))
    config.OUT_TABLAS = os.path.abspath(os.path.join(output_root, "tablas"))
    return os.path.abspath(output_root)


def resolver_casos(args, cfg):
    if args.case_dir:
        return [os.path.basename(os.path.abspath(args.case_dir))]
    if args.case_id:
        return list(dict.fromkeys(args.case_id))
    if args.cohort:
        cohort_path = args.cohort if os.path.isabs(args.cohort) else os.path.join(ROOT, args.cohort)
        df = pd.read_csv(cohort_path)
        if args.process_only and "process" in df.columns:
            df = df[df["process"].astype(str).isin({"1", "1.0", "True", "true"})]
        return df["case_id"].astype(str).tolist()
    if "case_ids" in cfg and cfg["case_ids"]:
        return [str(c) for c in cfg["case_ids"]]
    return config.detectar_casos()


def hash_config(cfg, data_root):
    payload = {
        "config": cfg,
        "data_root": os.path.abspath(data_root),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_preprocesamiento(data_root):
    payload = {
        "data_root": os.path.abspath(data_root),
        "modalities": list(config.MODALITIES_IMG),
        "n4_shrink": int(config.N4_SHRINK),
        "wiener_size": int(config.WIENER_SIZE),
        "normalization_scheme": str(config.NORM_SUSTRACCION),
        "spacing_objetivo": tuple(config.SPACING_OBJETIVO),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def marker_path(case_id):
    return os.path.join(config.OUT_SEG, case_id, ".done")


def leer_marker(case_id):
    path = marker_path(case_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def marker_valido(case_id, cfg_hash):
    marker = leer_marker(case_id)
    return bool(marker and marker.get("config_hash") == cfg_hash and marker.get("status") == "done")


def escribir_marker(case_id, cfg_hash, df_case):
    dst = os.path.join(config.OUT_SEG, case_id)
    os.makedirs(dst, exist_ok=True)
    metrics_path = os.path.join(dst, "metrics.csv")
    df_case.to_csv(metrics_path, index=False)
    marker = {
        "case_id": case_id,
        "status": "done",
        "config_hash": cfg_hash,
        "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "methods": df_case["metodo"].astype(str).tolist(),
        "metrics_csv": metrics_path,
    }
    tmp = marker_path(case_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, sort_keys=True)
    os.replace(tmp, marker_path(case_id))


def clean_marker_path(case_id):
    return os.path.join(config.OUT_LIMPIEZA, case_id, ".clean.done")


def leer_clean_marker(case_id):
    path = clean_marker_path(case_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def archivos_limpios_completos(case_id):
    expected = [f"{case_id}-{mod}.nii.gz" for mod in config.MODALITIES_IMG + ["seg"]]
    return all(os.path.exists(os.path.join(config.OUT_LIMPIEZA, case_id, name))
               for name in expected)


def clean_marker_valido(case_id, preproc_hash):
    marker = leer_clean_marker(case_id)
    return bool(marker and marker.get("preproc_hash") == preproc_hash
                and marker.get("status") == "done"
                and archivos_limpios_completos(case_id))


def escribir_clean_marker(case_id, preproc_hash):
    dst = os.path.join(config.OUT_LIMPIEZA, case_id)
    os.makedirs(dst, exist_ok=True)
    marker = {
        "case_id": case_id,
        "status": "done",
        "preproc_hash": preproc_hash,
        "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "files": [f"{case_id}-{mod}.nii.gz" for mod in config.MODALITIES_IMG + ["seg"]],
    }
    tmp = clean_marker_path(case_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, sort_keys=True)
    os.replace(tmp, clean_marker_path(case_id))


def escribir_metricas_agregadas(casos):
    frames = []
    for case_id in casos:
        metrics_path = os.path.join(config.OUT_SEG, case_id, "metrics.csv")
        if os.path.exists(metrics_path):
            frames.append(pd.read_csv(metrics_path))
    if not frames:
        return None
    df_all = pd.concat(frames, ignore_index=True)
    os.makedirs(config.OUT_TABLAS, exist_ok=True)
    csv_path = os.path.join(config.OUT_TABLAS, "metricas_ET.csv")
    df_all.to_csv(csv_path, index=False)
    return df_all


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def requirements_versions():
    path = os.path.join(ROOT, "requirements-lock.txt")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def escribir_provenance(output_root, cfg, cfg_hash, preproc_hash, casos,
                        run_cases, skipped_cases, started_at, ended_at,
                        phase_timings=None, clean_stats=None):
    df = escribir_metricas_agregadas(casos)
    per_case = {}
    if df is not None:
        for case_id, sub in df.groupby("case_id"):
            per_case[case_id] = {
                "case_wall_s": float(sub["case_wall_s"].iloc[0]) if "case_wall_s" in sub else None,
                "shared_preproc_s": float(sub["shared_preproc_s"].iloc[0]) if "shared_preproc_s" in sub else None,
                "methods": {
                    row.metodo: {
                        "tiempo_s": float(getattr(row, "tiempo_s", 0.0)),
                        "dice_ET": float(getattr(row, "dice_ET", 0.0)),
                        "vol_pred": int(getattr(row, "vol_pred", 0)),
                    }
                    for row in sub.itertuples(index=False)
                },
            }
    prov = {
        "started_at": started_at,
        "ended_at": ended_at,
        "command": sys.argv,
        "git_hash": git_hash(),
        "python": sys.version,
        "platform": platform.platform(),
        "requirements_lock": requirements_versions(),
        "resolved_config": cfg,
        "config_hash": cfg_hash,
        "preproc_hash": preproc_hash,
        "seed": config.SEED,
        "data_root": config.DATASET_DIR,
        "clean_root": config.OUT_LIMPIEZA,
        "output_root": os.path.abspath(output_root),
        "metrics_csv": os.path.join(config.OUT_TABLAS, "metricas_ET.csv"),
        "case_list": casos,
        "run_cases": run_cases,
        "skipped_cases": skipped_cases,
        "clean_run_cases": (clean_stats or {}).get("run", []),
        "clean_skipped_cases": (clean_stats or {}).get("skipped", []),
        "phase_timings_s": phase_timings or {},
        "per_case_timings": per_case,
    }
    prov_dir = os.path.join(output_root, "provenance")
    os.makedirs(prov_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(prov_dir, f"provenance_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
    latest = os.path.join(prov_dir, "latest_provenance.json")
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
    return path


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
def paso_limpieza(casos, preproc_hash):
    print("\n" + "="*60)
    print("PASO 1  LIMPIEZA  (Wiener + N4 + normalización percentil)")
    print("="*60)
    stats = {"run": [], "skipped": []}
    for case_id in casos:
        print(f"\n  [{case_id}]")
        if clean_marker_valido(case_id, preproc_hash):
            print("  [skip] limpieza vigente (.clean.done)")
            stats["skipped"].append(case_id)
            continue
        clean_pipeline.limpiar_caso(
            case_id,
            base_dir=config.DATASET_DIR,
            out_root=config.OUT_LIMPIEZA,
            mysize_wiener=config.WIENER_SIZE,
            n4_shrink=config.N4_SHRINK,
            esquema_norm=str(config.NORM_SUSTRACCION),
            verbose=True,
        )
        escribir_clean_marker(case_id, preproc_hash)
        stats["run"].append(case_id)
    print(f"\n  ✓ {len(casos)} caso(s) limpios en {config.OUT_LIMPIEZA}")


# ── PASO 2: Segmentación ───────────────────────────────────────────
    return stats


def paso_segmentacion(casos, args, resolved_config, cfg_hash):
    print("\n" + "="*60)
    print("PASO 2  SEGMENTACIÓN ET")
    print("="*60)

    methods = metodos_desde_config(resolved_config, legacy=args.legacy)
    if methods:
        print("  Metodos:", ", ".join(methods))

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

        t0 = time.perf_counter()
        mascaras, gt_et, df = correr_pipeline_et(
            t1c, t1n, seg,
            t1c_raw=t1c_raw,
            t1n_raw=t1n_raw,
            t2f=t2f,
            semilla_zyx=semilla,
            case_id=case_id,
            auto_pct=float(resolved_config.get("auto_pct", config.AUTO_PCT)),
            sigma=float(resolved_config.get("sigma", config.SIGMA)),
            methods=methods,
            verbose=True,
        )
        case_wall_s = time.perf_counter() - t0
        if "case_wall_s" not in df.columns:
            df["case_wall_s"] = round(float(case_wall_s), 4)
        todos_df.append(df)
        todos_mascaras[case_id] = mascaras

        # Guardar máscaras como NIfTI
        dst = os.path.join(config.OUT_SEG, case_id)
        os.makedirs(dst, exist_ok=True)
        for nombre, pred in mascaras.items():
            if nombre.startswith("_"): continue
            ps = io_utils.desde_numpy(pred.astype(np.uint8), ref=t1c)
            out_path = os.path.join(dst, f"{case_id}-et_{nombre}.nii.gz")
            tmp_path = os.path.join(dst, f"{case_id}-et_{nombre}.tmp.nii.gz")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            sitk.WriteImage(sitk.Cast(ps, sitk.sitkUInt8),
                            tmp_path, useCompression=True)
            os.replace(tmp_path, out_path)
        escribir_marker(case_id, cfg_hash, df)

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
    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    resolved_config = resolver_config(args)
    aplicar_config_runtime(resolved_config)
    output_root = configurar_rutas(args, resolved_config)
    config.asegurar_dirs()

    casos = resolver_casos(args, resolved_config)
    if not casos:
        print(f"\n✗  No hay casos en: {config.DATASET_DIR}")
        print("  Estructura esperada:")
        print("  images/BraTS-GLI-XXXXX-XXX/")
        print("    *-t1n.nii.gz  *-t1c.nii.gz  *-t2w.nii.gz  *-t2f.nii.gz  *-seg.nii.gz")
        sys.exit(1)

    cfg_hash = hash_config(resolved_config, config.DATASET_DIR)
    preproc_hash = hash_preprocesamiento(config.DATASET_DIR)
    skipped_cases = [] if args.force else [c for c in casos if marker_valido(c, cfg_hash)]
    skipped_set = set(skipped_cases)
    run_cases = [c for c in casos if c not in skipped_set]

    print(f"\nCasos seleccionados ({len(casos)}):")
    for c in casos:
        print(f"  • {c}")

    print(f"\nCheckpoint config_hash={cfg_hash[:16]}")
    print(f"  preproc_hash={preproc_hash[:16]}")
    print(f"  run={len(run_cases)}  skipped={len(skipped_cases)}")

    mascaras = {}
    clean_stats = {"run": [], "skipped": []}
    phase_timings = {"cleanup_s": 0.0, "segmentation_s": 0.0,
                     "visualization_s": 0.0, "poisson_s": 0.0}
    if run_cases and not args.skip_clean:
        t_phase = time.perf_counter()
        clean_stats = paso_limpieza(run_cases, preproc_hash)
        phase_timings["cleanup_s"] = round(time.perf_counter() - t_phase, 4)
    else: print("\n[skip] Limpieza.")

    if run_cases and not args.skip_seg:
        t_phase = time.perf_counter()
        mascaras = paso_segmentacion(run_cases, args, resolved_config, cfg_hash)
        escribir_metricas_agregadas(casos)
        phase_timings["segmentation_s"] = round(time.perf_counter() - t_phase, 4)
    elif not args.skip_seg:
        print("\n[skip] SegmentaciÃ³n: todos los casos tienen checkpoint vigente.")
        escribir_metricas_agregadas(casos)
    else: print("\n[skip] Segmentación.")

    if not args.skip_viz and mascaras:
        t_phase = time.perf_counter()
        paso_visualizacion(run_cases, mascaras)
        phase_timings["visualization_s"] = round(time.perf_counter() - t_phase, 4)
    else: print("\n[skip] Visualización.")

    if not args.skip_poisson and mascaras:
        t_phase = time.perf_counter()
        paso_poisson(run_cases, mascaras,
                     threshold=float(resolved_config.get("poisson_threshold",
                                                         config.POISSON_THRESHOLD)))
        phase_timings["poisson_s"] = round(time.perf_counter() - t_phase, 4)
    else: print("\n[skip] Poisson.")

    ended_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    provenance_path = escribir_provenance(
        output_root, resolved_config, cfg_hash, preproc_hash, casos,
        run_cases, skipped_cases, started_at, ended_at,
        phase_timings=phase_timings, clean_stats=clean_stats)

    print("\n" + "="*60)
    print("LISTO")
    print(f"  Provenance -> {provenance_path}")
    print(f"  Figuras  → {config.OUT_FIG}")
    print(f"  Métricas → {config.OUT_TABLAS}/metricas_ET.csv")
    print(f"  Máscaras → {config.OUT_SEG}")
    print("="*60)


if __name__ == "__main__":
    main()
