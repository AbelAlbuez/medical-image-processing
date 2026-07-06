# -*- coding: utf-8 -*-
"""
make_report_figs.py
===================
Genera las DOS figuras propias (contornos deformables + Poisson) que usa
informe_unificado.tex, con la paleta del informe, y las deja en figuras/:

  figuras/def_ranking.png            -> Dice-ET medio por método (barras)
  figuras/poisson_02108_chanvese.png -> superficie de Poisson GT vs predicción
"""
import os, shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "figuras"), exist_ok=True)
df = pd.read_csv(os.path.join(ROOT, "output", "tablas", "metricas_ET.csv"))
mean = df.groupby("metodo")["dice_ET"].mean()

# Métodos a mostrar: nuestros 4 (deformables) + 2 referencias por intensidad
filas = [
    ("variational_spline", "Chan–Vese (variacional)", "#553a86", True),
    ("bspline",            "B-spline",                "#7559ad", True),
    ("level_set",          "Level set (geodésico)",   "#3a78c2", True),
    ("spline",             "Snake (spline)",          "#7fb0e0", True),
    ("otsu_T1c",           "Otsu T1c (ref.)",         "#d6a23e", False),
    ("gmm_T1c",            "GMM T1c (ref.)",          "#e9cd86", False),
]
filas = [(m, lab, col, ours, float(mean[m])) for m, lab, col, ours in filas]
filas.sort(key=lambda r: r[4])            # ascendente -> el mejor arriba

labels = [r[1] for r in filas]
vals   = [r[4] for r in filas]
cols   = [r[2] for r in filas]

fig, ax = plt.subplots(figsize=(7.4, 3.8))
bars = ax.barh(labels, vals, color=cols, edgecolor="white", height=0.66)
ax.axvline(0.50, color="#c0392b", ls="--", lw=1.2, alpha=0.8)
ax.text(0.505, -0.6, "umbral 0,50", color="#c0392b", fontsize=9, va="center")
for b, v in zip(bars, vals):
    ax.text(v + 0.006, b.get_y() + b.get_height()/2, f"{v:.3f}".replace(".", ","),
            va="center", fontsize=10, fontweight="bold", color="#1e2a44")
ax.set_xlim(0, 0.62)
ax.set_xlabel("Dice-ET medio (20 casos ricos en ET)", fontsize=10)
ax.set_title("Segmentación del ET por contornos deformables\n"
             "(púrpura/azul = métodos nuevos · arena = referencia por intensidad)",
             fontsize=11, color="#13294b")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=9.5)
fig.tight_layout()
out1 = os.path.join(ROOT, "figuras", "def_ranking.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print("OK", out1)

# Poisson del mejor caso (Chan–Vese, 02108)
src = os.path.join(ROOT, "output", "figuras", "poisson",
                   "BraTS-GLI-02108-100_variational_spline_poisson.png")
out2 = os.path.join(ROOT, "figuras", "poisson_02108_chanvese.png")
if os.path.exists(src):
    shutil.copy(src, out2)
    print("OK", out2)
else:
    print("[!] no encontrado:", src)
