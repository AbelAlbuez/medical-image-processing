"""
run_experiments.py
==================
Ejecuta experimentos automaticos de segmentacion con Watershed y Level Sets
sobre todas las imagenes en ../images/.

Uso:
    python run_experiments.py                    # todos los experimentos
    python run_experiments.py --metodo watershed # solo watershed
    python run_experiments.py --metodo level_sets # solo level sets
    python run_experiments.py --imagen MRBrainTumor.nii.gz  # una sola imagen
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base — calculadas relativas a este script para que funcionen
# independientemente del directorio de trabajo
# ---------------------------------------------------------------------------
DIR_SCRIPTS = Path(__file__).parent.resolve()
DIR_BASE    = DIR_SCRIPTS.parent
DIR_IMAGENES = DIR_BASE / "images"
PYTHON      = sys.executable  # usa el interprete del venv activo

# ---------------------------------------------------------------------------
# Imagenes disponibles (se detectan automaticamente)
# ---------------------------------------------------------------------------
IMAGENES = sorted(DIR_IMAGENES.glob("*.nii.gz")) + sorted(DIR_IMAGENES.glob("*.nii"))

# ---------------------------------------------------------------------------
# Configuracion de experimentos: Watershed
#
# Se prueban distintas combinaciones de sigma, threshold y level para
# observar el efecto de cada parametro en la segmentacion.
#
# sigma     → controla la suavidad del gradiente gaussiano
# threshold → descarta gradientes muy bajos (ruido de fondo)
# level     → cuanto se inunda antes de detener la fusion de cuencas
# ---------------------------------------------------------------------------
EXPERIMENTOS_WATERSHED = [
    # Configuracion base
    {"sigma": 1.0,  "threshold": 0.01, "level": 0.2},
    # Sigma alto: gradiente mas suave, menos over-segmentacion
    {"sigma": 2.0,  "threshold": 0.01, "level": 0.2},
    # Level alto: mas fusion de cuencas → regiones mas grandes
    {"sigma": 1.0,  "threshold": 0.01, "level": 0.5},
    # Threshold alto: ignora mas ruido de borde
    {"sigma": 1.0,  "threshold": 0.05, "level": 0.2},
    # Combinacion agresiva: menos regiones, bordes suaves
    {"sigma": 2.0,  "threshold": 0.05, "level": 0.4},
]

# ---------------------------------------------------------------------------
# Configuracion de experimentos: Level Sets (Fast Marching)
#
# seed         → punto de inicio del frente de onda (None = centro del volumen)
# alpha        → pendiente de la sigmoide (negativo → invierte la curva)
# beta         → punto de inflexion de la sigmoide (ajustar segun la imagen)
# stopping_value → cuanta region cubre el frente antes de detenerse
# iterations   → iteraciones de difusion anisotropica (mas = mas lento)
# ---------------------------------------------------------------------------
EXPERIMENTOS_LEVEL_SETS = [
    # Configuracion base
    {
        "seed": None,
        "sigma": 1.0,
        "alpha": -0.5,
        "beta": 3.0,
        "stopping_value": 100.0,
        "iterations": 5,
        "initial_distance": 5.0,
    },
    # Mayor stopping_value → frente avanza mas lejos
    {
        "seed": None,
        "sigma": 1.0,
        "alpha": -0.5,
        "beta": 3.0,
        "stopping_value": 200.0,
        "iterations": 5,
        "initial_distance": 5.0,
    },
    # Mas iteraciones de difusion → imagen mas suave, bordes mas limpios
    {
        "seed": None,
        "sigma": 1.0,
        "alpha": -0.5,
        "beta": 3.0,
        "stopping_value": 100.0,
        "iterations": 15,
        "initial_distance": 5.0,
    },
    # Sigma mayor del gradiente → bordes menos sensibles al ruido
    {
        "seed": None,
        "sigma": 2.0,
        "alpha": -0.5,
        "beta": 3.0,
        "stopping_value": 100.0,
        "iterations": 5,
        "initial_distance": 5.0,
    },
]


def parsear_argumentos():
    parser = argparse.ArgumentParser(
        description="Ejecutor automatico de experimentos de segmentacion - Taller 3"
    )
    parser.add_argument(
        "--metodo",
        type=str,
        default="ambos",
        choices=["watershed", "level_sets", "ambos"],
        help="Metodo a ejecutar (default: ambos)",
    )
    parser.add_argument(
        "--imagen",
        type=str,
        default=None,
        help="Nombre de la imagen a procesar (ej: MRBrainTumor.nii.gz). "
             "Si no se especifica, procesa todas las imagenes disponibles.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Muestra los comandos que se ejecutarian sin ejecutarlos.",
    )
    return parser.parse_args()


def seleccionar_imagenes(nombre_filtro):
    """Devuelve la lista de rutas de imagenes a procesar."""
    if not IMAGENES:
        print(f"[ERROR] No se encontraron imagenes en: {DIR_IMAGENES}")
        sys.exit(1)

    if nombre_filtro:
        candidatos = [img for img in IMAGENES if img.name == nombre_filtro]
        if not candidatos:
            print(f"[ERROR] Imagen '{nombre_filtro}' no encontrada en {DIR_IMAGENES}")
            print("Imagenes disponibles:")
            for img in IMAGENES:
                print(f"  - {img.name}")
            sys.exit(1)
        return candidatos

    return IMAGENES


def ejecutar_comando(cmd, dry_run=False):
    """Ejecuta un comando de shell e imprime la salida en tiempo real."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n{'='*70}")
    print(f"CMD: {cmd_str}")
    print(f"{'='*70}")

    if dry_run:
        print("[DRY RUN] Comando no ejecutado.")
        return True

    t0 = time.perf_counter()
    try:
        resultado = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False,  # imprime en tiempo real
        )
        t1 = time.perf_counter()
        print(f"[OK] Completado en {t1 - t0:.2f} s")
        return True
    except subprocess.CalledProcessError as e:
        t1 = time.perf_counter()
        print(f"[ERROR] El proceso termino con codigo {e.returncode} ({t1 - t0:.2f} s)")
        return False


def correr_watershed(imagenes, dry_run=False):
    """Ejecuta todos los experimentos de Watershed."""
    script = DIR_SCRIPTS / "watershed.py"
    total = len(imagenes) * len(EXPERIMENTOS_WATERSHED)
    exitosos = 0
    fallidos  = 0

    print(f"\n{'#'*70}")
    print(f"# WATERSHED — {len(imagenes)} imagen(es) x {len(EXPERIMENTOS_WATERSHED)} configuraciones = {total} ejecuciones")
    print(f"{'#'*70}")

    for imagen in imagenes:
        for exp in EXPERIMENTOS_WATERSHED:
            cmd = [
                PYTHON, str(script),
                str(imagen),
                "--sigma",     str(exp["sigma"]),
                "--threshold", str(exp["threshold"]),
                "--level",     str(exp["level"]),
                "--output_dir", str(DIR_BASE / "output" / "watershed"),
            ]
            ok = ejecutar_comando(cmd, dry_run=dry_run)
            if ok:
                exitosos += 1
            else:
                fallidos += 1

    return exitosos, fallidos


def correr_level_sets(imagenes, dry_run=False):
    """Ejecuta todos los experimentos de Level Sets."""
    script = DIR_SCRIPTS / "level_sets.py"
    total = len(imagenes) * len(EXPERIMENTOS_LEVEL_SETS)
    exitosos = 0
    fallidos  = 0

    print(f"\n{'#'*70}")
    print(f"# LEVEL SETS — {len(imagenes)} imagen(es) x {len(EXPERIMENTOS_LEVEL_SETS)} configuraciones = {total} ejecuciones")
    print(f"{'#'*70}")

    for imagen in imagenes:
        for exp in EXPERIMENTOS_LEVEL_SETS:
            cmd = [
                PYTHON, str(script),
                str(imagen),
                "--sigma",          str(exp["sigma"]),
                "--alpha",          str(exp["alpha"]),
                "--beta",           str(exp["beta"]),
                "--stopping_value", str(exp["stopping_value"]),
                "--iterations",     str(exp["iterations"]),
                "--initial_distance", str(exp["initial_distance"]),
                "--output_dir",     str(DIR_BASE / "output" / "level_sets"),
            ]
            # Si hay semilla explicita, agregarla; si no, omitir (se calcula automaticamente)
            if exp["seed"] is not None:
                x, y, z = exp["seed"]
                cmd += ["--seed", str(x), str(y), str(z)]

            ok = ejecutar_comando(cmd, dry_run=dry_run)
            if ok:
                exitosos += 1
            else:
                fallidos += 1

    return exitosos, fallidos


def imprimir_resumen(resultados):
    """Imprime tabla de resumen al final."""
    print(f"\n{'='*70}")
    print("RESUMEN DE EJECUCION")
    print(f"{'='*70}")
    total_ok  = sum(r[0] for r in resultados.values())
    total_err = sum(r[1] for r in resultados.values())
    for metodo, (ok, err) in resultados.items():
        print(f"  {metodo:<15}: {ok} exitosos, {err} fallidos")
    print(f"  {'TOTAL':<15}: {total_ok} exitosos, {total_err} fallidos")
    print(f"{'='*70}\n")


def main():
    args = parsear_argumentos()

    imagenes = seleccionar_imagenes(args.imagen)
    print(f"\nImagenes seleccionadas:")
    for img in imagenes:
        print(f"  - {img.name}")

    t_global = time.perf_counter()
    resultados = {}

    if args.metodo in ("watershed", "ambos"):
        ok, err = correr_watershed(imagenes, dry_run=args.dry_run)
        resultados["Watershed"] = (ok, err)

    if args.metodo in ("level_sets", "ambos"):
        ok, err = correr_level_sets(imagenes, dry_run=args.dry_run)
        resultados["Level Sets"] = (ok, err)

    t_total = time.perf_counter() - t_global
    imprimir_resumen(resultados)
    print(f"Tiempo total de todos los experimentos: {t_total:.2f} s")


if __name__ == "__main__":
    main()
