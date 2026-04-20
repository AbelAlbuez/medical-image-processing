"""
otsu_segmentation.py
Taller 2 — Segmentación por Umbrales
Procesamiento de Imágenes Médicas — Pontificia Universidad Javeriana

Aplica OtsuMultipleThresholdsImageFilter con n = 1, 2, 3.
Genera:
  - results/otsu/<key>_otsu_<n>.nii.gz   → volúmenes segmentados
  - results/otsu/png/<key>_otsu_<n>.png  → vista detallada 3 vistas
  - results/otsu/png/<key>_resumen.png   → comparativa n=1,2,3

Referencia ITK:
  https://examples.itk.org/src/filtering/thresholding/thresholdanimageusingotsu/documentation
"""

import os
import itk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(SCRIPT_DIR, "..", "images")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results", "otsu")
PNG_DIR     = os.path.join(RESULTS_DIR, "png")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

IMAGES = {
    "brain":  os.path.join(IMAGES_DIR, "MRBrainTumor.nii.gz"),
    "breast": os.path.join(IMAGES_DIR, "MRBreastCancer.nii.gz"),
    "liver":  os.path.join(IMAGES_DIR, "MRLiverTumor.nii.gz"),
}

LABELS_ES = {
    "brain":  "Tumor Cerebral (MR)",
    "breast": "Cáncer de Mama (MR)",
    "liver":  "Tumor Hepático (MR)",
}

# Centroide del tumor por imagen (z, y, x) — verificado en 3D Slicer
TUMOR_CENTER = {
    "brain":  (89, 144, 131),
    "breast": ( 8, 271, 505),
    "liver":  (57,  69, 116),  # tumor: masa blanca esquina inferior derecha
}

# Para mama el volumen solo tiene 30 slices en Z → coronal y sagital
# salen como una línea delgada. Se usan 3 axiales (inf/central/sup) en su lugar.
USE_AXIALS_ONLY = {"breast"}

# Nombre descriptivo de cada etiqueta
TISSUE_NAMES = {
    "brain":  {0: "Fondo / CSF",    1: "Sustancia gris",    2: "Sustancia blanca", 3: "Tumor (hiper.)"},
    "breast": {0: "Fondo",          1: "Tejido oscuro",     2: "Tejido intermedio",3: "Tumor (hiper.)"},
    "liver":  {0: "Fondo",          1: "Tejido hepático",   2: "Tejido hiper.",    3: "Tumor (hiper.)"},
}

NUM_THRESHOLDS_LIST = [1, 2, 3]
NUM_HISTOGRAM_BINS  = 128

COLORS = ["#111111", "#29B6F6", "#66BB6A", "#FFA726"]

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _norm(sl):
    mn, mx = float(sl.min()), float(sl.max())
    return (sl - mn) / (mx - mn + 1e-9)


def _get_views(arr_orig, arr_seg, key):
    """
    Devuelve 3 pares (orig_slice, seg_slice, titulo).
    Brain/Liver: axial + coronal + sagital centradas en el tumor.
    Breast: 3 cortes axiales (inf / central / sup) del tumor.
    """
    cz, cy, cx = TUMOR_CENTER[key]

    if key in USE_AXIALS_ONLY:
        nz = arr_orig.shape[0]
        z_inf = max(0, cz - 3)
        z_sup = min(nz - 1, cz + 3)
        return [
            (arr_orig[z_inf, :, :], arr_seg[z_inf, :, :], f"Axial inf. z={z_inf}"),
            (arr_orig[cz,   :, :], arr_seg[cz,   :, :], f"Axial central z={cz} ★"),
            (arr_orig[z_sup, :, :], arr_seg[z_sup, :, :], f"Axial sup. z={z_sup}"),
        ]
    else:
        return [
            (arr_orig[cz, :, :],  arr_seg[cz, :, :],  f"Axial z={cz}"),
            (arr_orig[:, cy, :],  arr_seg[:, cy, :],  f"Coronal y={cy}"),
            (arr_orig[:, :, cx],  arr_seg[:, :, cx],  f"Sagital x={cx}"),
        ]


# ─────────────────────────────────────────────────────────────
# PNG vistas detalladas
# ─────────────────────────────────────────────────────────────

def guardar_vistas_png(key, n, arr_orig, arr_seg, umbrales, limites, ruta_png):
    views    = _get_views(arr_orig, arr_seg, key)
    n_clases = n + 1
    cmap_sub = ListedColormap(COLORS[:n_clases])
    tissue   = TISSUE_NAMES[key]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.patch.set_facecolor("#0D0D0D")
    fig.suptitle(f"{LABELS_ES[key]}  —  Otsu  n={n}  |  umbrales: {umbrales}",
                 color="white", fontsize=12, fontweight="bold", y=0.98)

    for col, (orig_sl, seg_sl, titulo) in enumerate(views):
        for row in range(2):
            ax = axes[row, col]
            ax.set_facecolor("#000000")
            ax.axis("off")
            ax.imshow(_norm(orig_sl), cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
            if row == 1:
                seg_m = np.ma.masked_where(seg_sl == 0, seg_sl)
                ax.imshow(seg_m, cmap=cmap_sub, vmin=0, vmax=n,
                          alpha=0.65, origin="lower", interpolation="nearest", aspect="auto")
            if row == 0:
                color_t = "#FFD600" if "★" in titulo else "#AAAAAA"
                ax.set_title(titulo, color=color_t, fontsize=8)
            if col == 0:
                ax.set_ylabel("Original" if row == 0 else "Otsu", color="white", fontsize=8)

    patches = []
    for et in range(n_clases):
        name  = tissue.get(et, f"Et.{et}")
        pct   = float(np.sum(arr_seg == et)) / arr_seg.size * 100
        lo, hi = limites[et], limites[et + 1]
        arrow = " ←" if et == n else ""
        patches.append(mpatches.Patch(color=COLORS[et],
            label=f"Et.{et}  {name}{arrow}  [{lo:.0f}–{hi:.0f}]  {pct:.1f}%"))

    fig.legend(handles=patches, loc="lower center", ncol=n_clases,
               facecolor="#1A1A1A", edgecolor="#444444",
               labelcolor="white", fontsize=8, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    plt.savefig(ruta_png, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    PNG vistas: {os.path.basename(ruta_png)}")


# ─────────────────────────────────────────────────────────────
# PNG resumen (formato referencia)
# ─────────────────────────────────────────────────────────────

def guardar_resumen_png(key, arr_orig, resultados, ruta_png):
    tissue   = TISSUE_NAMES[key]
    cz, cy, cx = TUMOR_CENTER[key]

    # Column definitions depend on image type
    if key in USE_AXIALS_ONLY:
        nz = arr_orig.shape[0]
        z_inf = max(0, cz - 3)
        z_sup = min(nz - 1, cz + 3)
        col_defs = [
            (lambda a, s: (a[z_inf, :, :], s[z_inf, :, :]), f"Axial inf. {z_inf}",    False),
            (lambda a, s: (a[cz,   :, :], s[cz,   :, :]), f"Axial central {cz}",    True),
            (lambda a, s: (a[z_sup, :, :], s[z_sup, :, :]), f"Axial sup. {z_sup}",    False),
        ]
    else:
        col_defs = [
            (lambda a, s: (a[cz, :, :],  s[cz, :, :]),  f"Axial z={cz}",     False),
            (lambda a, s: (a[:, cy, :],  s[:, cy, :]),  f"Coronal y={cy}",   True),
            (lambda a, s: (a[:, :, cx],  s[:, :, cx]),  f"Sagital x={cx}",   False),
        ]

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("#0D0D0D")
    fig.suptitle(f"Otsu — {LABELS_ES[key]}  |  Comparativa n=1, 2, 3",
                 color="white", fontsize=14, fontweight="bold", y=0.995)

    gs = GridSpec(3, 9, figure=fig,
                  left=0.01, right=0.99, top=0.95, bottom=0.02,
                  wspace=0.03, hspace=0.06,
                  width_ratios=[1, 1, 0.04, 1, 1, 0.04, 1, 1, 1.55])

    vista_cols = [(0, 1), (3, 4), (6, 7)]
    axes_grid  = {}

    for row, (n, arr_seg, umbrales, limites) in enumerate(resultados):
        n_clases = n + 1
        cmap_sub = ListedColormap(COLORS[:n_clases])

        for vista_idx, (slicer_fn, col_title, is_central) in enumerate(col_defs):
            orig_sl, seg_sl = slicer_fn(arr_orig, arr_seg)
            c_orig, c_otsu  = vista_cols[vista_idx]

            ax_o = fig.add_subplot(gs[row, c_orig])
            ax_s = fig.add_subplot(gs[row, c_otsu])
            axes_grid[(row, vista_idx)] = ax_o

            ax_o.imshow(_norm(orig_sl), cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
            ax_o.axis("off"); ax_o.set_facecolor("#000000")

            ax_s.imshow(_norm(orig_sl), cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
            seg_m = np.ma.masked_where(seg_sl == 0, seg_sl)
            ax_s.imshow(seg_m, cmap=cmap_sub, vmin=0, vmax=n,
                        alpha=0.70, origin="lower", interpolation="nearest", aspect="auto")
            ax_s.axis("off"); ax_s.set_facecolor("#000000")

            # Línea amarilla entre orig y otsu
            pos_o = ax_o.get_position()
            pos_s = ax_s.get_position()
            x_line = (pos_o.x1 + pos_s.x0) / 2
            fig.add_artist(plt.Line2D([x_line, x_line], [pos_s.y0, pos_s.y1],
                                      color="#FFD600", linewidth=1.2,
                                      transform=fig.transFigure, clip_on=False))

            # Títulos columna (solo fila 0)
            if row == 0:
                color_t = "#FFD600" if is_central else "#AAAAAA"
                star    = " ★" if is_central else ""
                mid_x   = (pos_o.x0 + pos_s.x1) / 2
                top_y   = pos_o.y1 + 0.008
                fig.text(mid_x, top_y, col_title + star,
                         ha="center", va="bottom", fontsize=8,
                         color=color_t, transform=fig.transFigure)

            # Labels orig/otsu (solo última fila)
            if row == 2:
                ax_o.text(0.5, -0.025, "original", transform=ax_o.transAxes,
                          ha="center", va="top", fontsize=6, color="#777777")
                ax_s.text(0.5, -0.025, "otsu", transform=ax_s.transAxes,
                          ha="center", va="top", fontsize=6, color="#777777")

        # Parámetros de fila
        ax_first = axes_grid[(row, 0)]
        txt = f"n={n}\n[{', '.join(str(u) for u in umbrales)}]\nEt.tumor={n}"
        ax_first.text(0.02, 0.98, txt, transform=ax_first.transAxes,
                      va="top", ha="left", fontsize=6.5, color="#DDDDDD",
                      fontfamily="monospace",
                      bbox=dict(facecolor="#00000099", edgecolor="none", pad=1.5))

    # ── Leyenda ──────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[:, 8])
    ax_leg.set_facecolor("#0D0D0D"); ax_leg.axis("off")

    ax_leg.text(0.06, 0.99, "LEYENDA", transform=ax_leg.transAxes,
                va="top", ha="left", fontsize=11, color="white", fontweight="bold")

    last_n, last_seg, last_umbrales, last_limites = resultados[-1]
    n_clases_max = last_n + 1

    ax_leg.text(0.06, 0.93, f"Etiquetas (n={last_n})", transform=ax_leg.transAxes,
                va="top", ha="left", fontsize=8, color="#AAAAAA")

    y_pos = 0.86
    for et in range(n_clases_max):
        name      = tissue.get(et, f"Et.{et}")
        pct       = float(np.sum(last_seg == et)) / last_seg.size * 100
        is_tumor  = (et == last_n)
        color_txt = "#FFD600" if is_tumor else "white"
        weight    = "bold"    if is_tumor else "normal"
        arrow     = " ◄"     if is_tumor else ""

        rect = mpatches.FancyBboxPatch(
            (0.05, y_pos - 0.042), 0.16, 0.038,
            boxstyle="round,pad=0.004",
            facecolor=COLORS[et], edgecolor="#555555", linewidth=0.5,
            transform=ax_leg.transAxes)
        ax_leg.add_patch(rect)

        ax_leg.text(0.26, y_pos - 0.004, f"Et.{et}  {name}{arrow}",
                    transform=ax_leg.transAxes, va="center", ha="left",
                    fontsize=8.5, color=color_txt, fontweight=weight)
        ax_leg.text(0.26, y_pos - 0.030, f"{pct:.1f}% del volumen",
                    transform=ax_leg.transAxes, va="center", ha="left",
                    fontsize=7, color="#AAAAAA")
        y_pos -= 0.13

    ax_leg.text(0.06, y_pos - 0.04,
                "Cada celda:\nizq = original\nder = Otsu\n\nLínea amarilla\ndivide ambas",
                transform=ax_leg.transAxes,
                va="top", ha="left", fontsize=7.5, color="#777777")

    plt.savefig(ruta_png, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Resumen PNG: {os.path.basename(ruta_png)}")


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

def aplicar_otsu(key, path):
    print(f"\n{'='*60}")
    print(f"  {LABELS_ES[key].upper()}")
    print(f"{'='*60}")

    imagen = itk.imread(path, itk.F)
    arr    = itk.array_from_image(imagen)
    IT     = type(imagen)
    tissue = TISSUE_NAMES[key]

    print(f"  Shape : {arr.shape}")
    print(f"  Rango : min={arr.min():.1f}  max={arr.max():.1f}")
    nz = arr[arr > 0]
    print(f"  P50={np.percentile(nz,50):.1f}  P90={np.percentile(nz,90):.1f}  P99={np.percentile(nz,99):.1f}")

    resultados = []

    for n in NUM_THRESHOLDS_LIST:
        filtro = itk.OtsuMultipleThresholdsImageFilter[IT, IT].New()
        filtro.SetInput(imagen)
        filtro.SetNumberOfThresholds(n)
        filtro.SetNumberOfHistogramBins(NUM_HISTOGRAM_BINS)
        filtro.Update()

        umbrales  = [round(float(u), 2) for u in filtro.GetThresholds()]
        resultado = filtro.GetOutput()
        arr_seg   = itk.array_from_image(resultado)
        limites   = [0.0] + umbrales + [float(arr.max())]

        print(f"\n  n={n}  umbrales={umbrales}")
        for et in range(n + 1):
            cnt    = int(np.sum(arr_seg == et))
            pct    = cnt / arr_seg.size * 100
            name   = tissue.get(et, f"Et.{et}")
            marker = " ← TUMOR" if et == n else ""
            print(f"    Et.{et} ({name}): {cnt:>9,} vóx ({pct:5.2f}%)  "
                  f"[{limites[et]:.1f}–{limites[et+1]:.1f}]{marker}")

        ruta_nii = os.path.join(RESULTS_DIR, f"{key}_otsu_{n}.nii.gz")
        itk.imwrite(resultado, ruta_nii)
        print(f"    NII: {os.path.basename(ruta_nii)}")

        ruta_png = os.path.join(PNG_DIR, f"{key}_otsu_{n}.png")
        guardar_vistas_png(key, n, arr, arr_seg, umbrales, limites, ruta_png)

        resultados.append((n, arr_seg, umbrales, limites))

    ruta_resumen = os.path.join(PNG_DIR, f"{key}_resumen.png")
    guardar_resumen_png(key, arr, resultados, ruta_resumen)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SEGMENTACIÓN — MÉTODO DE OTSU")
    print("  Taller 2 | Procesamiento de Imágenes Médicas")
    print("  Pontificia Universidad Javeriana — 2026")
    print("="*60)

    for key, path in IMAGES.items():
        if not os.path.exists(path):
            print(f"\n  ⚠  No encontrado: {path}")
            continue
        aplicar_otsu(key, path)

    print("\n" + "="*60)
    print("  ✓  Completado — NII en results/otsu/  |  PNG en results/otsu/png/")
    print("="*60)