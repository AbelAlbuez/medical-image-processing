"""
config.py
=========
Rutas y constantes compartidas por todo el pipeline.

Para cambiar de entorno solo hay que tocar DATASET_DIR / PROJECT_ROOT,
o exportar las variables de entorno equivalentes antes de correr.
"""
from __future__ import annotations
import os

# --------------------------------------------------------------------------- #
# Modalidades y etiquetas
# --------------------------------------------------------------------------- #
MODALITIES_IMG = ["t1n", "t1c", "t2w", "t2f"]
SUFIJOS_SALIDA = MODALITIES_IMG + ["seg"]

LABEL_MAP = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
LABEL_ET  = 3

SUBREGIONES_BRATS = {
    "WT": (1, 2, 3, 4),
    "TC": (1, 3, 4),
    "ET": (3,),
}

BIMODALIDAD_MIN = 5.0 / 9.0
SEED = int(os.environ.get("BRATS_SEED", "0"))

# WARNING: Stage 3A guard repair. These thresholds are pinned by the frozen
# regression fixture, including boundary case BraTS-GLI-02116-100. They allow a
# seed-divergent deformable mask only when the evolved contour is compact,
# enhancing, and not a volume blow-up. Changing them changes study numerics.
ENABLE_EVIDENCE_GUARD = os.environ.get("BRATS_ENABLE_EVIDENCE_GUARD", "1") not in {"0", "false", "False"}
GUARD_MIN_LCC_FRACTION = float(os.environ.get("BRATS_GUARD_MIN_LCC_FRACTION", "0.90"))
GUARD_MIN_ENHANCEMENT_RATIO = float(os.environ.get("BRATS_GUARD_MIN_ENHANCEMENT_RATIO", "0.90"))
GUARD_MAX_VOLUME_MULTIPLE = float(os.environ.get("BRATS_GUARD_MAX_VOLUME_MULTIPLE", "2.50"))

# WARNING: Stage 3B best-iterate selection was rejected for the study baseline.
# OFF preserves the iter-N behavior. Do not enable it in default configs unless
# the sacred-case gate and frozen baselines are intentionally re-run.
ENABLE_BEST_ITERATE = os.environ.get("BRATS_ENABLE_BEST_ITERATE", "0") not in {"0", "false", "False"}
BEST_ITERATE_W_LCC = float(os.environ.get("BRATS_BEST_ITERATE_W_LCC", "1.0"))
BEST_ITERATE_W_ENHANCEMENT = float(os.environ.get("BRATS_BEST_ITERATE_W_ENHANCEMENT", "1.0"))
BEST_ITERATE_W_VOLUME_STABILITY = float(os.environ.get("BRATS_BEST_ITERATE_W_VOLUME_STABILITY", "1.0"))
BEST_ITERATE_PATIENCE = int(os.environ.get("BRATS_BEST_ITERATE_PATIENCE", "8"))

# Method parameters surfaced through configs/pipeline.yaml by run_all.py.
AUTO_PCT = float(os.environ.get("BRATS_AUTO_PCT", "90.0"))
SIGMA = float(os.environ.get("BRATS_SIGMA", "0.5"))
POISSON_THRESHOLD = float(os.environ.get("BRATS_POISSON_THRESHOLD", "0.75"))
FAST_MARCHING_TIME_THRESHOLD = float(os.environ.get("BRATS_FAST_MARCHING_TIME_THRESHOLD", "35.0"))
VARIATIONAL_SPLINE_ITERS = int(os.environ.get("BRATS_VARIATIONAL_SPLINE_ITERS", "35"))
VARIATIONAL_SPLINE_SMOOTHING = int(os.environ.get("BRATS_VARIATIONAL_SPLINE_SMOOTHING", "3"))
BSPLINE_CHANVESE_ITERS = int(os.environ.get("BRATS_BSPLINE_CHANVESE_ITERS", "35"))
BSPLINE_CHANVESE_SMOOTHING = int(os.environ.get("BRATS_BSPLINE_CHANVESE_SMOOTHING", "2"))
LEVEL_SET_ITERS = int(os.environ.get("BRATS_LEVEL_SET_ITERS", "120"))
N4_SHRINK = int(os.environ.get("BRATS_N4_SHRINK", "4"))
WIENER_SIZE = int(os.environ.get("BRATS_WIENER_SIZE", "3"))

# Esquema de normalización por tarea
NORM_CLASICA     = "zscore"
NORM_SUSTRACCION = "percentil"

# --------------------------------------------------------------------------- #
# Rutas  —  solo ajustar estas dos variables (o exportar las env vars)
# --------------------------------------------------------------------------- #
# Donde están las imágenes: PROJECT_ROOT/images/<case_id>/<case_id>-t1n.nii.gz ...
PROJECT_ROOT = os.environ.get("BRATS_PROJECT_ROOT",
                              os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                           "..", "..", "..")))

DATASET_DIR  = os.environ.get("BRATS_DATASET_DIR",
                              os.path.join(PROJECT_ROOT, "images"))

# Salidas
OUT_LIMPIEZA = os.path.join(PROJECT_ROOT, "output", "limpieza")
OUT_SEG      = os.path.join(PROJECT_ROOT, "output", "segmentacion")
OUT_FIG      = os.path.join(PROJECT_ROOT, "output", "figuras")
OUT_TABLAS   = os.path.join(PROJECT_ROOT, "output", "tablas")

# CSVs del EDA (opcionales; si no existen los métodos recalculan todo)
CSV_DIR       = os.environ.get("BRATS_CSV_DIR", PROJECT_ROOT)
CSV_CASOS     = os.path.join(CSV_DIR, "casos_demostrativos.csv")
CSV_OTSU      = os.path.join(CSV_DIR, "EDA_intensidad_otsu.csv")
CSV_SEED      = os.path.join(CSV_DIR, "EDA_intensidad_seed.csv")
CSV_STATS     = os.path.join(CSV_DIR, "EDA_intensidad_stats.csv")

SPACING_OBJETIVO = (1.0, 1.0, 1.0)


def asegurar_dirs() -> None:
    for d in (OUT_LIMPIEZA, OUT_SEG, OUT_FIG, OUT_TABLAS):
        os.makedirs(d, exist_ok=True)


def detectar_casos(base_dir: str = None) -> list:
    """
    Detecta automáticamente todos los casos en `base_dir` (= DATASET_DIR).
    Un caso válido es una subcarpeta que contiene al menos un archivo .nii.gz.
    """
    base_dir = base_dir or DATASET_DIR
    if not os.path.isdir(base_dir):
        return []
    casos = []
    for nombre in sorted(os.listdir(base_dir)):
        ruta = os.path.join(base_dir, nombre)
        if os.path.isdir(ruta):
            niis = [f for f in os.listdir(ruta) if f.endswith(".nii.gz")]
            if niis:
                casos.append(nombre)
    return casos
