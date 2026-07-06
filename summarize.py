"""Resumen de métricas ET: media de Dice por método y nº de casos > umbral."""
import os, sys
import pandas as pd

ROOT = os.environ.get("BRATS_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
csv = os.path.join(ROOT, "output", "tablas", "metricas_ET.csv")
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75

df = pd.read_csv(csv)
g = df.groupby("metodo")["dice_ET"]
res = pd.DataFrame({
    "dice_medio": g.mean().round(3),
    "dice_mediana": g.median().round(3),
    "dice_max": g.max().round(3),
    f"casos>{thr}": df[df.dice_ET > thr].groupby("metodo")["dice_ET"].count(),
}).fillna(0)
res[f"casos>{thr}"] = res[f"casos>{thr}"].astype(int)
res = res.sort_values("dice_medio", ascending=False)
print(f"\n=== Resumen Dice-ET ({df.case_id.nunique()} casos) ===\n")
print(res.to_string())
print(f"\nTotal (caso,método) con Dice > {thr}: {(df.dice_ET > thr).sum()}")
