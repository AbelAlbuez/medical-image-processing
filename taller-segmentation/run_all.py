#!/usr/bin/env python
"""
run_all.py
Taller 2 — Segmentación por Umbrales
Ejecuta el pipeline completo: venv → dependencias → todos los scripts
Uso: python run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).parent.resolve()
VENV_DIR    = RAIZ / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
VENV_PIP    = VENV_DIR / "bin" / "pip"

SCRIPTS = [
    "scripts/histogram.py",
    "scripts/binary_threshold.py",
    "scripts/otsu_segmentation.py",
    "scripts/kmeans_segmentation.py",
    "scripts/visualize_results.py",
]

LOGS_DIR = RAIZ / "logs"

# ---------------------------------------------------------------------------
# Directorios de salida que deben existir antes de ejecutar los scripts
# ---------------------------------------------------------------------------

DIRS_REQUERIDOS = [
    LOGS_DIR,
    RAIZ / "results" / "binary_threshold",
    RAIZ / "results" / "otsu",
    RAIZ / "results" / "kmeans",
    RAIZ / "report" / "figures",
]

# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------


def crear_venv() -> None:
    """Crea el entorno virtual si no existe."""
    if VENV_DIR.exists():
        print("[SETUP] Entorno virtual encontrado en venv/")
        return

    print("[SETUP] Creando entorno virtual en venv/ ...")
    resultado = subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        cwd=str(RAIZ),
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        print(f"[SETUP] ERROR al crear el entorno virtual:\n{resultado.stderr}")
        sys.exit(1)
    print("[SETUP] Entorno virtual creado correctamente")


def instalar_dependencias() -> None:
    """Actualiza pip e instala las dependencias desde requirements.txt."""
    print("[SETUP] Actualizando pip ...")
    resultado = subprocess.run(
        [str(VENV_PIP), "install", "--upgrade", "pip"],
        cwd=str(RAIZ),
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        print(f"[SETUP] ERROR al actualizar pip:\n{resultado.stderr}")
        sys.exit(1)

    print("[SETUP] Instalando dependencias desde requirements.txt ...")
    resultado = subprocess.run(
        [str(VENV_PIP), "install", "-r", str(RAIZ / "requirements.txt")],
        cwd=str(RAIZ),
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        print(f"[SETUP] ERROR al instalar dependencias:\n{resultado.stderr}")
        sys.exit(1)
    print("[SETUP] Dependencias instaladas correctamente")


def guardar_log(nombre: str, stdout: str, stderr: str, returncode: int) -> None:
    """
    Guarda un archivo de log para el script ejecutado.

    Parámetros
    ----------
    nombre     : str — nombre del script (sin ruta, ej: histogram.py)
    stdout     : str — salida estándar del script
    stderr     : str — salida de error del script
    returncode : int — código de retorno
    """
    log_name = Path(nombre).stem + ".log"
    log_path = LOGS_DIR / log_name

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Script:    {nombre}\n")
        f.write(f"Returncode: {returncode}\n")
        f.write(f"\n{'='*60}\n  STDOUT\n{'='*60}\n")
        f.write(stdout if stdout else "(vacío)\n")
        f.write(f"\n{'='*60}\n  STDERR\n{'='*60}\n")
        f.write(stderr if stderr else "(vacío)\n")


def ejecutar_script(script_path: str) -> bool:
    """
    Ejecuta un script con el Python del venv y maneja su resultado.

    Parámetros
    ----------
    script_path : str — ruta relativa al script (ej: scripts/histogram.py)

    Retorna
    -------
    bool — True si el script terminó exitosamente, False en caso contrario
    """
    nombre = Path(script_path).name
    print(f"\n{'─'*60}")
    print(f"  Ejecutando: {nombre}")
    print(f"{'─'*60}")

    resultado = subprocess.run(
        [str(VENV_PYTHON), str(RAIZ / script_path)],
        cwd=str(RAIZ),
        capture_output=True,
        text=True,
    )

    # Guardar log independientemente del resultado
    guardar_log(nombre, resultado.stdout, resultado.stderr, resultado.returncode)

    if resultado.returncode != 0:
        print(f"\n  ✗ {nombre} FALLÓ (returncode={resultado.returncode})")
        if resultado.stderr:
            print(f"\n  STDERR:\n{resultado.stderr}")
        return False

    # Imprimir salida del script
    if resultado.stdout:
        print(resultado.stdout)
    print(f"  ✓ {nombre} completado")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("="*60)
    print("  TALLER 2 — SEGMENTACIÓN POR UMBRALES")
    print("  Iniciando pipeline completo")
    print("="*60)

    # Crear directorios de salida
    for d in DIRS_REQUERIDOS:
        d.mkdir(parents=True, exist_ok=True)

    # Paso 1 — Entorno virtual
    crear_venv()

    # Paso 2 — Dependencias
    instalar_dependencias()

    # Paso 3 — Ejecutar scripts en orden
    print("\n[PIPELINE] Ejecutando scripts de procesamiento ...\n")
    for script in SCRIPTS:
        exito = ejecutar_script(script)
        if not exito:
            nombre = Path(script).name
            print(f"\n  ✗ Pipeline detenido por fallo en: {nombre}")
            print(f"  Revise el log en: logs/{Path(nombre).stem}.log")
            sys.exit(1)

    # Resumen final
    print(f"\n{'='*42}")
    print("  TALLER 2 — SEGMENTACIÓN POR UMBRALES")
    print("  Ejecución completada exitosamente")
    print(f"{'='*42}")
    print("Scripts ejecutados:")
    for script in SCRIPTS:
        print(f"  ✓ {Path(script).name}")
    print()
    print("Resultados generados en:")
    print("  results/binary_threshold/")
    print("  results/otsu/")
    print("  results/kmeans/")
    print("  report/figures/")
    print("  logs/")
    print(f"{'='*42}")


if __name__ == "__main__":
    main()
