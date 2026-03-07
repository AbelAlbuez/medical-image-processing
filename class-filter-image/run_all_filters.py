#!/usr/bin/env python
"""
Runner de experimentos: ejecuta los filtros en src/ sobre las imágenes en samples/
mediante subprocess (CLI). No modifica ni importa la lógica de los scripts en src.
"""

import os
import subprocess
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuración (rutas relativas al directorio del script / raíz del proyecto)
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
SAMPLES_DIR = PROJECT_ROOT / "samples"
SRC_DIR = PROJECT_ROOT / "src"
RESULT_BASE = PROJECT_ROOT / "result"

# Parámetros experimentales sugeridos (ver FILTER_ANALYSIS.md)
MEAN_RADII = [2, 3, 4]
MEDIAN_RADII = [2, 3, 4]
HISTOGRAM_ALPHAS = [0.3, 0.5]
HISTOGRAM_BETAS = [0.3, 0.5]
HISTOGRAM_RADII = [3, 5]

# Carpetas de salida por filtro
OUTPUT_DIRS = {
    "gradient": RESULT_BASE / "gradient_results",
    "mean": RESULT_BASE / "mean_results",
    "median": RESULT_BASE / "median_results",
    "adaptive_histogram": RESULT_BASE / "adaptive_histogram_results",
}


def get_sample_basename(path: Path) -> str:
    """Nombre base del archivo sin .nii ni .nii.gz."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]  # len(".nii.gz") == 7
    if name.endswith(".nii"):
        return name[:-4]
    return name


def list_sample_images() -> list[Path]:
    """Lista de archivos en samples/ con extensión .nii o .nii.gz."""
    if not SAMPLES_DIR.is_dir():
        return []
    out = []
    for p in sorted(SAMPLES_DIR.iterdir()):
        if p.is_file() and (
            p.name.endswith(".nii.gz") or p.name.endswith(".nii")
        ):
            out.append(p)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], label: str) -> tuple[bool, str]:
    """Ejecuta comando por subprocess. Retorna (éxito, mensaje_error)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip() or "Unknown error"
            return False, err
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def run_gradient(input_path: Path, output_path: Path) -> tuple[bool, str]:
    script = SRC_DIR / "gradient.py"
    return run_cmd(
        [sys.executable, str(script), str(input_path), str(output_path)],
        "gradient",
    )


def run_mean(input_path: Path, output_path: Path, radius: int) -> tuple[bool, str]:
    script = SRC_DIR / "mean.py"
    return run_cmd(
        [sys.executable, str(script), str(input_path), str(output_path), str(radius)],
        "mean",
    )


def run_median(input_path: Path, output_path: Path, radius: int) -> tuple[bool, str]:
    script = SRC_DIR / "median.py"
    return run_cmd(
        [sys.executable, str(script), str(input_path), str(output_path), str(radius)],
        "median",
    )


def run_adaptive_histogram(
    input_path: Path, output_path: Path, alpha: float, beta: float, radius: int
) -> tuple[bool, str]:
    script = SRC_DIR / "histogram.py"
    return run_cmd(
        [
            sys.executable,
            str(script),
            str(input_path),
            str(output_path),
            str(alpha),
            str(beta),
            str(radius),
        ],
        "adaptive_histogram",
    )


def main() -> None:
    ensure_dir(RESULT_BASE)
    for d in OUTPUT_DIRS.values():
        ensure_dir(d)

    samples = list_sample_images()
    if not samples:
        print(f"No se encontraron imágenes en {SAMPLES_DIR}")
        print("Extensiones esperadas: .nii, .nii.gz")
        sys.exit(1)

    print(f"Imágenes a procesar: {[p.name for p in samples]}")
    print(f"Resultados en: {RESULT_BASE}")
    print()

    success_count = 0
    fail_count = 0
    errors: list[str] = []

    # --- Gradient (sin parámetros) ---
    for inp in samples:
        base = get_sample_basename(inp)
        out_path = OUTPUT_DIRS["gradient"] / f"{base}_gradient.nii"
        print(f"  [gradient] {inp.name} -> {out_path.name}")
        ok, err = run_gradient(inp, out_path)
        if ok:
            success_count += 1
        else:
            fail_count += 1
            msg = f"gradient | {inp.name} | {err}"
            errors.append(msg)
            print(f"    ERROR: {err}")

    # --- Mean (radius) ---
    for inp in samples:
        base = get_sample_basename(inp)
        for r in MEAN_RADII:
            out_path = OUTPUT_DIRS["mean"] / f"{base}_mean_r{r}.nii"
            print(f"  [mean r={r}] {inp.name} -> {out_path.name}")
            ok, err = run_mean(inp, out_path, r)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                errors.append(f"mean r={r} | {inp.name} | {err}")
                print(f"    ERROR: {err}")

    # --- Median (radius) ---
    for inp in samples:
        base = get_sample_basename(inp)
        for r in MEDIAN_RADII:
            out_path = OUTPUT_DIRS["median"] / f"{base}_median_r{r}.nii"
            print(f"  [median r={r}] {inp.name} -> {out_path.name}")
            ok, err = run_median(inp, out_path, r)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                errors.append(f"median r={r} | {inp.name} | {err}")
                print(f"    ERROR: {err}")

    # --- Adaptive histogram (alpha, beta, radius) ---
    for inp in samples:
        base = get_sample_basename(inp)
        for alpha in HISTOGRAM_ALPHAS:
            for beta in HISTOGRAM_BETAS:
                for r in HISTOGRAM_RADII:
                    out_path = (
                        OUTPUT_DIRS["adaptive_histogram"]
                        / f"{base}_hist_a{alpha}_b{beta}_r{r}.nii"
                    )
                    print(f"  [hist a={alpha} b={beta} r={r}] {inp.name} -> {out_path.name}")
                    ok, err = run_adaptive_histogram(inp, out_path, alpha, beta, r)
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                        errors.append(
                            f"hist a={alpha} b={beta} r={r} | {inp.name} | {err}"
                        )
                        print(f"    ERROR: {err}")

    # --- Resumen ---
    print()
    print("---------- Resumen ----------")
    print(f"  Éxitos: {success_count}")
    print(f"  Fallos: {fail_count}")
    if errors:
        print("  Detalle de fallos:")
        for e in errors:
            print(f"    - {e}")
    print("------------------------------")


if __name__ == "__main__":
    main()
