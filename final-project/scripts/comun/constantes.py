"""
constantes.py
=============
Constantes y rutas compartidas por todo el pipeline. Centralizamos aquí TODO lo que
pueda cambiar (rutas, etiquetas, parámetros globales) para que los 5 módulos no tengan
"números mágicos" ni rutas hardcodeadas.

Rutas SIEMPRE con pathlib.Path, ancladas a la raíz del proyecto (final-project/), que se
deduce de la ubicación de este archivo: scripts/comun/constantes.py -> parents[2].
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto (deducidas, NO hardcodeadas)
# --------------------------------------------------------------------------- #
# scripts/comun/constantes.py -> parents[0]=comun, parents[1]=scripts, parents[2]=final-project
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

# ZIPs de datos (NO se descomprimen completos; se leen con zipfile).
ZIP_TRAINING: Path = DATASETS_DIR / "BraTS2024-BraTS-GLI-TrainingData.zip"            # con seg (GT). PRINCIPAL.
ZIP_ADICIONAL: Path = DATASETS_DIR / "BraTS2024-BraTS-GLI-AdditionalTrainingData.zip"  # reserva (con seg).
ZIP_VALIDATION: Path = DATASETS_DIR / "BraTS2024-BraTS-GLI-ValidationData.zip"         # SIN etiquetas.

# Carpeta temporal para extraer puntualmente caso+modalidad desde el ZIP.
TMP_DIR: Path = OUTPUT_DIR / "_tmp_nii"

# --------------------------------------------------------------------------- #
# Modalidades y etiquetas BraTS 2024 GLI (copiadas de config.py de Santiago)
# --------------------------------------------------------------------------- #
MODALIDADES_IMG = ["t1n", "t1c", "t2w", "t2f"]   # modalidades de imagen (sin seg)
SUFIJOS = MODALIDADES_IMG + ["seg"]              # todos los sufijos de archivo por caso

# NUESTRO objetivo: el Tumor Realzante (ET) sobre T1c.
MODALIDAD_OBJETIVO = "t1c"   # toda la segmentación de ET vive sobre T1c
LABEL_ET = 3                 # ET = label 3 en BraTS 2024 GLI

# BraTS-GLI post-tratamiento: 4 etiquetas (copiado de Santiago).
LABEL_MAP = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}

# Sub-regiones "anidadas" estándar de BraTS (copiado de Santiago). Nuestro foco es ET.
#   WT (Whole Tumor) = {1,2,3,4}   TC (Tumor Core) = {1,3,4}   ET (Enhancing) = {3}
SUBREGIONES_BRATS = {
    "WT": (1, 2, 3, 4),
    "TC": (1, 3, 4),
    "ET": (3,),
}

# Umbral del coeficiente de bimodalidad (copiado de Santiago). Por encima => bimodal y
# Otsu fiable; por debajo => unimodal y Otsu tiende a fallar. 5/9 ~= 0.5556 (uniforme).
BIMODALIDAD_MIN = 5.0 / 9.0

# --------------------------------------------------------------------------- #
# Geometría esperada (validada en el EDA: el GLI viene 1mm iso, LAS, coregistrado)
# --------------------------------------------------------------------------- #
SPACING_OBJETIVO = (1.0, 1.0, 1.0)

# --------------------------------------------------------------------------- #
# Parámetros globales de ejecución
# --------------------------------------------------------------------------- #
MAX_CASOS_DEFAULT = 100   # subconjunto a procesar; configurable por --max_casos
SEED = 42                 # semilla global para reproducibilidad (KMeans/GMM/muestreo)

# Carpetas de salida por módulo (cada una con outputs/ + figuras/ + <modulo>_reporte.html).
SALIDAS_MODULO = {
    "eda":           OUTPUT_DIR / "eda",
    "limpieza":      OUTPUT_DIR / "limpieza",
    "registro":      OUTPUT_DIR / "registro",
    "segmentacion":  OUTPUT_DIR / "segmentacion",
    "visualizacion": OUTPUT_DIR / "visualizacion",
}


def dirs_modulo(modulo: str) -> dict[str, Path]:
    """
    Devuelve {'base','outputs','figuras','reporte'} para un módulo y crea las carpetas.
    `reporte` lleva NOMBRE PROPIO del módulo (p. ej. eda_reporte.html), nunca report.html.
    """
    base = SALIDAS_MODULO[modulo]
    outputs = base / "outputs"
    figuras = base / "figuras"
    outputs.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "outputs": outputs,
        "figuras": figuras,
        "reporte": base / f"{modulo}_reporte.html",
    }


# --------------------------------------------------------------------------- #
# Paleta y estilo de los reportes HTML (idénticos entre módulos)
# --------------------------------------------------------------------------- #
PALETA = {
    "azul":    "#1f77b4",
    "naranja": "#ff7f0e",
    "morado":  "#9467bd",
    "verde":   "#2ca02c",
    "rojo":    "#d62728",
    "tinta":   "#1f2a37",
    "suave":   "#5b6b7c",
    "linea":   "#e3e8ef",
    "fondo":   "#f6f8fb",
    "hero_a":  "#1e3c72",   # degradado hero (inicio)
    "hero_b":  "#3a7bd5",   # degradado hero (fin)
    "tabla_cabecera": "#eef2f8",
}

# Color para marcar la semilla en figuras (rojo, por convención del proyecto).
COLOR_SEMILLA = PALETA["rojo"]
