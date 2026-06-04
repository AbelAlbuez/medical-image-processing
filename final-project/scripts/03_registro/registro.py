#!/usr/bin/env python3
"""
Módulo 3 — REGISTRO demostrativo (Mattes MI + Euler3D rígido, multi-resolución)
===============================================================================
Origen: NUESTRO notebook notebooks/Registro_BraTS2024_GLI.ipynb (Santiago no tiene
registro). Pasado a script ejecutable independiente, reutilizando comun/.

Modo demostrativo por caso (lee casos_demostrativos de parametros_eda.json):
  1. T1c como volumen FIJO (fixed).
  2. Se aplica una perturbación rígida CONOCIDA (rot (6,4,-5)°, tras (5,-4,3) mm) ->
     volumen "desalineado" (moving).
  3. Se registra moving->fixed para RECUPERAR la transformación.
  4. Como la perturbación es conocida, el TRE residual debe ser SUB-VÓXEL (<1 mm).

Motor (SimpleITK):
  Mattes MI 50 bins · muestreo aleatorio 0.1 (seed=42) · 150 iter · shrink [4,2] ·
  sigmas [2,1] · CenteredTransformInitializer (GEOMETRY) · Gradient Descent Line Search.

Produce (en output/registro/):
  outputs/metricas_registro.csv            (una fila por caso + fila PROMEDIO)
  outputs/parametros_usados.json
  outputs/transformaciones/<caso>-transform.tfm
  outputs/<caso>-desalineado.nii.gz  y  <caso>-recuperado.nii.gz
  figuras/registro_<caso>.png  y  convergencia_registro.png
  registro_reporte.html
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import reporte as R              # noqa: E402

import matplotlib.pyplot as plt             # noqa: E402

# --------------------------------------------------------------------------- #
# Configuración del registro (del Taller 4, idéntica al notebook)
# --------------------------------------------------------------------------- #
NBINS_MI = 50
MUESTREO = 0.1                # fracción de vóxeles muestreados por iteración
ITERS    = 150
SHRINK   = [4, 2]            # multi-resolución hasta 2 mm
SIGMAS   = [2, 1]

# Perturbación rígida CONOCIDA (ground truth del modo demostrativo).
ROT_DEG  = (6.0, 4.0, -5.0)   # grados (x, y, z)
TRA_MM   = (5.0, -4.0, 3.0)   # mm (x, y, z)


# --------------------------------------------------------------------------- #
# Motor de registro (copiado de nuestro notebook)
# --------------------------------------------------------------------------- #
def perturbar(fixed: sitk.Image):
    """Aplica la transformación rígida conocida y devuelve (transform, volumen_desalineado)."""
    centro = fixed.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in fixed.GetSize()])
    p = sitk.Euler3DTransform()
    p.SetCenter(centro)
    p.SetRotation(*[np.deg2rad(d) for d in ROT_DEG])
    p.SetTranslation(TRA_MM)
    moving = sitk.Resample(fixed, fixed, p, sitk.sitkLinear, 0.0)
    return p, moving


def registrar(fixed: sitk.Image, moving: sitk.Image):
    """Registro rígido por información mutua (Mattes). Devuelve transform, curva, MI, iteraciones."""
    Rm = sitk.ImageRegistrationMethod()
    Rm.SetMetricAsMattesMutualInformation(numberOfHistogramBins=NBINS_MI)
    Rm.SetMetricSamplingStrategy(Rm.RANDOM)
    Rm.SetMetricSamplingPercentage(MUESTREO, seed=C.SEED)
    Rm.SetInterpolator(sitk.sitkLinear)
    Rm.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0, numberOfIterations=ITERS,
        convergenceMinimumValue=1e-7, convergenceWindowSize=20)
    Rm.SetOptimizerScalesFromPhysicalShift()
    ini = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    Rm.SetInitialTransform(ini, inPlace=False)
    Rm.SetShrinkFactorsPerLevel(SHRINK)
    Rm.SetSmoothingSigmasPerLevel(SIGMAS)
    Rm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    curva = []
    Rm.AddCommand(sitk.sitkIterationEvent, lambda: curva.append(Rm.GetMetricValue()))
    final = Rm.Execute(fixed, moving)
    return final, curva, Rm.GetMetricValue(), Rm.GetOptimizerIteration()


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def dice_cerebro(a: sitk.Image, b: sitk.Image) -> float:
    A = sitk.GetArrayFromImage(a) > 0
    B = sitk.GetArrayFromImage(b) > 0
    s = A.sum() + B.sum()
    return float(2 * (A & B).sum() / s) if s else 0.0


def mse(a: sitk.Image, b: sitk.Image) -> float:
    A = sitk.GetArrayFromImage(a)
    B = sitk.GetArrayFromImage(b)
    return float(np.mean((A - B) ** 2))


def tre_mm(fixed: sitk.Image, perturb: sitk.Transform, final: sitk.Transform) -> float:
    """
    TRE residual sobre puntos de control distribuidos en el volumen. Compone la
    perturbación con la transform recuperada: si el registro recupera bien, la
    composición ~ identidad y el TRE ~ 0.
    """
    S = fixed.GetSize()
    fracs = [(.25, .25, .25), (.75, .25, .5), (.25, .75, .5),
             (.5, .5, .75), (.75, .75, .25), (.5, .25, .25)]
    idxs = [[S[0] * f0, S[1] * f1, S[2] * f2] for f0, f1, f2 in fracs]
    pts = [fixed.TransformContinuousIndexToPhysicalPoint([float(x) for x in idx]) for idx in idxs]
    return float(np.mean([
        np.linalg.norm(np.array(final.TransformPoint(perturb.TransformPoint(p))) - np.array(p))
        for p in pts]))


def _axial_medio(img: sitk.Image) -> np.ndarray:
    a = sitk.GetArrayFromImage(img)
    return a[a.shape[0] // 2]


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def figura_caso(cid: str, fixed, moving, recuperado, figuras: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (t, im) in zip(axes, [("Fija", fixed), ("Desalineada", moving), ("Recuperada", recuperado)]):
        ax.imshow(_axial_medio(im), cmap="gray", origin="lower", aspect="auto")
        ax.set_title(t); ax.axis("off")
    fig.suptitle(f"Registro — {cid} · T1c (axial central)")
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / f"registro_{cid}.png")


def figura_convergencia(curvas: dict, figuras: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cid, c in curvas.items():
        ax.plot(range(len(c)), c, label=cid, linewidth=1.6)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Mattes MI (menor = mejor)")
    ax.set_title("Convergencia del registro")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return R.guardar_figura(fig, figuras / "convergencia_registro.png")


# --------------------------------------------------------------------------- #
# Reporte HTML
# --------------------------------------------------------------------------- #
def construir_reporte(df, figs_caso, fig_conv, ruta_html):
    tre_min = float(df["TRE_mm"].min())
    dice_med = float(df["dice_cerebro_despues"].mean())
    kpis = [
        R.kpi(f"{tre_min:.3f} mm", "mejor TRE (residual)"),
        R.kpi(str(len(df)), "casos registrados"),
        R.kpi(f"{dice_med:.3f}", "Dice cerebro medio (después)"),
        R.kpi("T1c", "modalidad fija"),
    ]
    motor = pd.DataFrame([
        {"parametro": "Métrica", "valor": "Mattes Mutual Information"},
        {"parametro": "Transformación", "valor": "Euler3D (rígida: 3 rot + 3 tras)"},
        {"parametro": "Bins MI", "valor": NBINS_MI},
        {"parametro": "Muestreo", "valor": f"{MUESTREO} (aleatorio, seed={C.SEED})"},
        {"parametro": "Iteraciones máx", "valor": ITERS},
        {"parametro": "Shrink / Sigmas", "valor": f"{SHRINK} / {SIGMAS}"},
        {"parametro": "Inicializador", "valor": "CenteredTransformInitializer (GEOMETRY)"},
        {"parametro": "Optimizador", "valor": "Gradient Descent Line Search"},
        {"parametro": "Perturbación conocida", "valor": f"rot {ROT_DEG}° · tras {TRA_MM} mm"},
    ])
    secciones = []
    secciones.append(R.seccion(
        "Modo demostrativo",
        "<p>T1c se usa como volumen <b>fijo</b>. Se le aplica una perturbación rígida "
        f"<b>conocida</b> (rot {ROT_DEG}°, tras {TRA_MM} mm) para generar el volumen "
        "<b>desalineado</b>, y luego se registra <code>moving→fixed</code> para "
        "<b>recuperar</b> la transformación. Como la perturbación es conocida, el TRE "
        "residual mide directamente la calidad: debe ser <b>sub-vóxel (&lt;1 mm)</b>.</p>"))
    secciones.append(R.seccion("Motor de registro", R.df_a_tabla_html(motor)))
    secciones.append(R.seccion("Métricas por caso", R.df_a_tabla_html(df)))
    secciones.append(R.seccion(
        "Convergencia",
        R.tarjeta_figura(R.png_a_base64(fig_conv),
                         "Métrica Mattes MI por iteración (debe descender y estabilizarse).")))
    bloques = [R.tarjeta_figura(R.png_a_base64(p), f"{cid}: fija / desalineada / recuperada.")
               for cid, p in figs_caso.items()]
    secciones.append(R.seccion("Fija / desalineada / recuperada por caso", *bloques))
    R.armar_reporte(
        "Registro — T1c · BraTS 2024 GLI",
        kpis, secciones,
        subtitulo="Módulo 3 · Mattes Mutual Information + Euler3D rígido (multi-resolución)",
        ruta_salida=ruta_html)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    dirs = C.dirs_modulo("registro")
    tfm_dir = dirs["outputs"] / "transformaciones"
    tfm_dir.mkdir(parents=True, exist_ok=True)

    params_path = C.SALIDAS_MODULO["eda"] / "outputs" / "parametros_eda.json"
    if not params_path.exists():
        print(f"[ERROR] No existe {params_path}. Ejecuta primero el módulo 1 (EDA).")
        sys.exit(1)
    casos = json.loads(params_path.read_text(encoding="utf-8"))["casos_demostrativos"]
    print(f"[REGISTRO] casos demostrativos: {casos}\n")

    filas, curvas, figs_caso = [], {}, {}
    for i, cid in enumerate(casos, 1):
        fixed = io_zip.leer_sitk(cid, "t1c")
        if fixed is None:
            print(f"  [{i}/{len(casos)}] {cid}  -> SIN T1c, omitido")
            continue
        print(f"  [{i}/{len(casos)}] {cid}  registrando (Mattes MI + Euler3D)...")

        perturb, moving = perturbar(fixed)
        dice_antes, mse_antes = dice_cerebro(fixed, moving), mse(fixed, moving)
        t0 = time.time()
        final, curva, metrica, n_it = registrar(fixed, moving)
        recuperado = sitk.Resample(moving, fixed, final, sitk.sitkLinear, 0.0)
        seg = round(time.time() - t0, 1)
        curvas[cid] = curva

        fila = {
            "case_id": cid,
            "TRE_mm": round(tre_mm(fixed, perturb, final), 4),
            "dice_cerebro_antes": round(dice_antes, 4),
            "dice_cerebro_despues": round(dice_cerebro(fixed, recuperado), 4),
            "mse_antes": round(mse_antes, 5),
            "mse_despues": round(mse(fixed, recuperado), 5),
            "metrica_MI_final": round(float(metrica), 5),
            "iteraciones": int(n_it),
            "segundos": seg,
        }
        filas.append(fila)

        io_zip.guardar_sitk(moving, dirs["outputs"] / f"{cid}-desalineado.nii.gz")
        io_zip.guardar_sitk(recuperado, dirs["outputs"] / f"{cid}-recuperado.nii.gz")
        sitk.WriteTransform(final, str(tfm_dir / f"{cid}-transform.tfm"))
        figs_caso[cid] = figura_caso(cid, fixed, moving, recuperado, dirs["figuras"])

        print(f"        TRE={fila['TRE_mm']} mm  Dice {fila['dice_cerebro_antes']}→"
              f"{fila['dice_cerebro_despues']}  MI={fila['metrica_MI_final']}  {seg}s")

        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    if not filas:
        print("\n[ERROR] No se registró ningún caso.")
        sys.exit(1)

    df = pd.DataFrame(filas)
    # Fila PROMEDIO (solo columnas numéricas).
    prom = df.drop(columns=["case_id"]).mean(numeric_only=True).round(4).to_dict()
    prom["case_id"] = "PROMEDIO"
    df_out = pd.concat([df, pd.DataFrame([prom])[df.columns]], ignore_index=True)
    df_out.to_csv(dirs["outputs"] / "metricas_registro.csv", index=False)

    with open(dirs["outputs"] / "parametros_usados.json", "w", encoding="utf-8") as f:
        json.dump({
            "modalidad": "t1c", "modo": "demostrativo (perturbación conocida -> recuperación)",
            "NBINS_MI": NBINS_MI, "MUESTREO": MUESTREO, "seed": C.SEED, "ITERS": ITERS,
            "SHRINK": SHRINK, "SIGMAS": SIGMAS, "ROT_DEG": list(ROT_DEG), "TRA_MM": list(TRA_MM),
            "inicializador": "CenteredTransformInitializer GEOMETRY",
            "optimizador": "GradientDescentLineSearch",
        }, f, ensure_ascii=False, indent=2)

    fig_conv = figura_convergencia(curvas, dirs["figuras"])
    print("\n[REGISTRO] generando reporte HTML...")
    construir_reporte(df, figs_caso, fig_conv, dirs["reporte"])

    print("\n========== RESUMEN REGISTRO ==========")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(df_out.to_string(index=False))
    print(f"\n  salidas en : {dirs['base']}")
    print("======================================")


if __name__ == "__main__":
    main()
