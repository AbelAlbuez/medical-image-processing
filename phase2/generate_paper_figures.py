"""Generate paper/email figures from existing Phase 2 result artifacts.

This script reads only existing CSVs and reconstructed surface point/raster
outputs. It does not rerun segmentation, metrics, atlas construction, shape
filters, persistence, or surface reconstruction.

Outputs:
    phase2/figures/*.png
    phase2/figures/CAPTIONS.md
    phase2/brats_figures_bundle.zip
"""

from __future__ import annotations

from pathlib import Path
import math
import re
import shutil
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase2" / "figures"
ZIP_PATH = ROOT / "phase2" / "brats_figures_bundle.zip"
FINAL_DATA = ROOT / "phase2" / "final_data"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
METHOD_LABELS = {
    "otsu_T1c": "Otsu T1c",
    "gmm_T1c": "GMM T1c",
    "sustraccion": "Subtraction",
    "gmm_2d": "GMM 2D",
    "variational_spline": "Variational spline",
}
PALETTE = {
    "otsu_T1c": "#0072B2",
    "gmm_T1c": "#56B4E9",
    "sustraccion": "#E69F00",
    "gmm_2d": "#D55E00",
    "variational_spline": "#009E73",
    "R1": "#0072B2",
    "R2": "#CC79A7",
    "R3": "#009E73",
    "R4": "#D55E00",
}
PHASE_LABELS = {
    "baseline": "R1\nbaseline",
    "p1_spatial": "R2\nspatial",
    "p2b_soft_shape": "R3\nshape",
    "p3_cubical_persistence": "R4\nPH",
}
VOL_ORDER = ["absent", "small", "medium", "large"]
RNG = np.random.default_rng(0)


def read_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        final_path = FINAL_DATA / Path(rel).name
        if final_path.exists():
            path = final_path
    return pd.read_csv(path)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("EMPTY_NO_VALID_OPERATING_POINT", np.nan), errors="coerce")


def bootstrap_ci(values: pd.Series, stat: str = "mean", n: int = 4000) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    if stat == "median":
        center = float(np.median(arr))
        fn = np.median
    else:
        center = float(np.mean(arr))
        fn = np.mean
    reps = np.empty(n, dtype=float)
    for i in range(n):
        reps[i] = float(fn(RNG.choice(arr, size=arr.size, replace=True)))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return center, float(lo), float(hi)


def err_from_ci(center: float, lo: float, hi: float) -> np.ndarray:
    return np.array([[max(0.0, center - lo)], [max(0.0, hi - center)]])


def set_clean_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def savefig(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_impossibility_chain() -> Path:
    diag = read_csv("phase2/final_data/four_regime_discrimination_diagnostic.csv")
    rows = []
    for _, r in diag.iterrows():
        rid = r["regime_id"]
        val = str(r["separability_value"])
        if rid == "R1":
            score = 0.0
            note = "no intensity veto\n2/2 bright confounds"
        elif rid == "R2":
            nums = [float(x) for x in re.findall(r"\d+\.\d+", val)]
            score = 0.0
            note = f"confounds at\n{np.mean(nums):.1f}th atlas pct"
        elif rid == "R3":
            score = float(re.search(r"AUC\s+(\d+\.\d+)", val).group(1))
            note = "offline AUC 0.999\nnot deployable"
        else:
            score = float(re.search(r"AUC\s+(\d+\.\d+)", val).group(1))
            note = "inverted:\nconfounds more persistent"
        rows.append((rid, r["regime_label"].replace("R1 ", "").replace("R2 ", "").replace("R3 ", "").replace("R4 ", ""), score, note, r["separability_verdict"]))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(rows))
    colors = [PALETTE[r[0]] for r in rows]
    heights = [r[2] for r in rows]
    bars = ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.8)
    bars[2].set_hatch("//")
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1, label="random AUC")
    ax.axhline(0.75, color="#444444", linestyle=":", linewidth=1, label="useful separation target")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Tumor > confound separability score")
    ax.set_xticks(x)
    ax.set_xticklabels([r[1].replace(" PRIOR", "").replace(" BASELINE", "") for r in rows], rotation=15, ha="right")
    ax.set_title("Impossibility chain: enhancement-scalar regimes do not separate hard confounds")
    for i, (rid, _, score, note, verdict) in enumerate(rows):
        ax.text(i, min(score + 0.055, 1.03), f"{score:.3f}" if rid in {"R3", "R4"} else "no safe\nthreshold",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        if rid == "R4":
            continue
        ax.text(i, 0.08 if score > 0.2 else 0.17, note, ha="center", va="bottom", fontsize=8)
    ax.text(3, 0.36, "P3 inversion\nAUC 0.126", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDE0D2", ec=PALETTE["R4"], lw=1.2))
    ax.legend(frameon=False, loc="upper left")
    set_clean_axis(ax)
    return savefig(fig, "01_impossibility_chain_four_regimes.png")


def figure_growth_metric() -> Path:
    growth = read_csv("phase2/final_data/growth_metric_table.csv")
    phases = ["baseline", "p1_spatial", "p2b_soft_shape", "p3_cubical_persistence"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    for method in METHODS:
        flood = [to_num(growth.loc[growth["method"].eq(method), f"{p}_absent_flood_rate"]).iloc[0] for p in phases]
        detect = [to_num(growth.loc[growth["method"].eq(method), f"{p}_large_lesionwise_dice"]).iloc[0] for p in phases]
        axes[0].plot(range(len(phases)), flood, marker="o", color=PALETTE[method], label=METHOD_LABELS[method], linewidth=2)
        axes[1].plot(range(len(phases)), detect, marker="o", color=PALETTE[method], label=METHOD_LABELS[method], linewidth=2)
    axes[0].set_title("False-positive axis")
    axes[0].set_ylabel("Absent-case flood rate")
    axes[0].set_ylim(-0.04, 1.05)
    axes[1].set_title("Detection axis")
    axes[1].set_ylabel("Large-stratum lesion-wise Dice")
    axes[1].set_ylim(-0.02, 0.22)
    for ax in axes:
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels([PHASE_LABELS[p] for p in phases])
        set_clean_axis(ax)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Method trajectories across regimes on the locked Stage-4 metric", y=1.04)
    return savefig(fig, "02_growth_metric_method_trajectories.png")


def figure_best_worst() -> Path:
    case = read_csv("phase2/final_data/stage4_case_metrics.csv")
    surface = read_csv("phase2/final_data/surface_prediction_fidelity_cases.csv")
    methods = ["variational_spline", "gmm_2d"]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [PALETTE[m] for m in methods]

    panels = []
    for m in methods:
        d = case[(case["metodo"].eq(m)) & (case["vol_bin"].eq("large"))]["lesionwise_dice_mean"]
        fp = case[(case["metodo"].eq(m)) & (case["vol_bin"].eq("absent"))]["pred_vox"]
        asd = surface[(surface["method"].eq(m)) & (surface["vol_bin"].eq("large"))]["surface_asd"]
        panels.append({
            "dice": bootstrap_ci(d, "mean"),
            "fp": bootstrap_ci(fp, "median"),
            "asd": bootstrap_ci(asd, "median"),
        })

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    specs = [
        ("dice", "Large lesion-wise Dice", "mean", "higher is better"),
        ("fp", "Absent FP volume", "median voxels", "lower is better"),
        ("asd", "Large surface ASD", "median mm", "lower is better"),
    ]
    for ax, (key, title, ylabel, subtitle) in zip(axes, specs):
        centers = [p[key][0] for p in panels]
        lows = [p[key][1] for p in panels]
        highs = [p[key][2] for p in panels]
        xpos = np.arange(len(methods))
        yerr = np.vstack([np.array(centers) - np.array(lows), np.array(highs) - np.array(centers)])
        ax.bar(xpos, centers, color=colors, edgecolor="black", linewidth=0.8, yerr=yerr, capsize=4)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(f"{title}\n({subtitle})")
        ax.set_ylabel(ylabel)
        set_clean_axis(ax)
    fig.suptitle("Best detector vs worst baseline method: overlap, FP burden, and surface fidelity", y=1.04)
    return savefig(fig, "03_best_vs_worst_variational_spline_vs_gmm2d.png")


def figure_per_stratum() -> Path:
    present = read_csv("phase2/final_data/stage4_present_by_vol_bin.csv")
    surf = read_csv("phase2/final_data/surface_prediction_fidelity_by_method_stratum.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    x = np.arange(len(VOL_ORDER))
    width = 0.14
    for i, method in enumerate(METHODS):
        vals = []
        asd = []
        for vol in VOL_ORDER:
            if vol == "absent":
                vals.append(np.nan)
                asd.append(np.nan)
            else:
                sub = present[(present["metodo"].eq(method)) & (present["vol_bin"].eq(vol))]
                vals.append(float(sub["lesionwise_dice_mean"].iloc[0]) if len(sub) else np.nan)
                ss = surf[(surf["method"].eq(method)) & (surf["vol_bin"].eq(vol))]
                asd.append(float(ss["surface_asd_median"].iloc[0]) if len(ss) else np.nan)
        offset = (i - 2) * width
        axes[0].bar(x + offset, vals, width=width, color=PALETTE[method], label=METHOD_LABELS[method])
        axes[1].bar(x + offset, asd, width=width, color=PALETTE[method], label=METHOD_LABELS[method])
    for ax in axes:
        ax.axvspan(-0.5, 0.5, color="#eeeeee", zorder=-1)
        ax.text(0, ax.get_ylim()[1] * 0.85, "N/A for\nabsent GT", ha="center", va="top", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["absent\n(FP only)", "small", "medium", "large"])
        set_clean_axis(ax)
    axes[0].set_ylabel("FP-aware lesion-wise Dice")
    axes[0].set_title("Present-case detection by volume stratum")
    axes[1].set_ylabel("Surface ASD median (mm)")
    axes[1].set_title("Surface distance by volume stratum")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    fig.suptitle("Performance is stratum-dependent; absent cases require FP metrics", y=1.02)
    return savefig(fig, "04_per_stratum_lesionwise_and_surface_asd.png")


def figure_absent_flood() -> Path:
    absent = read_csv("phase2/final_data/stage4_absent_fp_summary.csv")
    absent = absent.set_index("metodo").loc[METHODS].reset_index()
    labels = [METHOD_LABELS[m] for m in absent["metodo"]]
    colors = [PALETTE[m] for m in absent["metodo"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    x = np.arange(len(absent))
    axes[0].bar(x, absent["flood_gt_10000_rate"], color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Flood rate on 33 absent cases")
    axes[0].set_title("Nearly all methods hallucinate ET on clean/absent cases")
    for xi, v in zip(x, absent["flood_gt_10000_rate"]):
        axes[0].text(xi, v + 0.025, f"{100*v:.0f}%", ha="center", fontsize=9)
    axes[1].bar(x, absent["fp_volume_median"], color=colors, edgecolor="black", linewidth=0.8)
    axes[1].set_ylabel("Median FP volume (voxels)")
    axes[1].set_title("False-positive burden differs even when flood is common")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        set_clean_axis(ax)
    fig.suptitle("Absent-case false-positive burden is a first-class result", y=1.04)
    return savefig(fig, "05_absent_case_flood_and_fp_volume.png")


def downsample_points(points: np.ndarray, n: int = 3500) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = RNG.choice(len(points), size=n, replace=False)
    return points[idx]


def load_points(case_id: str, method: str) -> np.ndarray:
    path = ROOT / "phase2" / "surface_reconstruction" / "surface_points" / "cohort" / case_id / f"{method}_poisson_points.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)["points"].astype(float)


def plot_cloud(ax, points: np.ndarray, color: str, title: str, lims: tuple[np.ndarray, np.ndarray]) -> None:
    pts = downsample_points(points)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=color, alpha=0.55, depthshade=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(lims[0][0], lims[1][0])
    ax.set_ylim(lims[0][1], lims[1][1])
    ax.set_zlim(lims[0][2], lims[1][2])
    ax.view_init(elev=18, azim=-62)
    ax.set_axis_off()


def figure_surface_renders() -> Path:
    good_case = "BraTS-GLI-02306-100"
    gt = load_points(good_case, "GT_ET")
    pred = load_points(good_case, "variational_spline")
    both = np.vstack([gt, pred])
    mins = both.min(axis=0)
    maxs = both.max(axis=0)
    pad = (maxs - mins).max() * 0.08
    lims = (mins - pad, maxs + pad)

    fig = plt.figure(figsize=(12, 8.6))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_cloud(ax1, gt, "#666666", "Good large case 02306: GT surface", lims)
    plot_cloud(ax2, pred, PALETTE["variational_spline"], "02306: variational_spline surface", lims)

    existing = [
        ROOT / "phase2" / "surface_reconstruction" / "figures" / "BraTS-GLI-00533-100_irreducible_surface_overlay.png",
        ROOT / "phase2" / "surface_reconstruction" / "figures" / "BraTS-GLI-02078-100_irreducible_surface_overlay.png",
    ]
    for idx, img_path in enumerate(existing, start=3):
        ax = fig.add_subplot(2, 2, idx)
        if img_path.exists():
            ax.imshow(mpimg.imread(img_path))
            ax.set_title(img_path.stem.replace("_", " "), fontsize=10)
        else:
            ax.text(0.5, 0.5, f"Missing render:\n{img_path.name}", ha="center", va="center")
        ax.axis("off")
    fig.suptitle("Poisson surfaces: good large case versus irreducible confound cases", y=0.98)
    return savefig(fig, "06_poisson_surface_renders_good_vs_irreducible.png")


def figure_surface_vs_dice() -> Path:
    surf = read_csv("phase2/final_data/surface_prediction_fidelity_by_method_stratum.csv")
    large = surf[surf["vol_bin"].eq("large")].copy()
    rho, p = spearmanr(large["lesionwise_dice_mean"], large["surface_asd_median"])
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for _, r in large.iterrows():
        method = r["method"]
        ax.scatter(r["lesionwise_dice_mean"], r["surface_asd_median"], s=95,
                   color=PALETTE[method], edgecolor="black", linewidth=0.8)
        ax.text(r["lesionwise_dice_mean"] + 0.003, r["surface_asd_median"] + 0.25,
                METHOD_LABELS[method], fontsize=8)
    ax.set_xlabel("Large-stratum lesion-wise Dice")
    ax.set_ylabel("Large-stratum surface ASD median (mm)")
    ax.set_title("Surface distance confirms the voxel metric on large tumors")
    ax.text(0.03, 0.95, f"Spearman rho = {rho:.2f}\np = {p:.3g}", transform=ax.transAxes,
            ha="left", va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999"))
    set_clean_axis(ax)
    return savefig(fig, "07_surface_vs_dice_large_stratum.png")


def write_captions(paths: list[Path]) -> Path:
    captions = {
        "01_impossibility_chain_four_regimes.png": (
            "Impossibility chain across the four methodological regimes. Intensity and location provide no safe separator for the hard confounds, shape proxies separate components offline but do not deploy cleanly, and genuine cubical H0 persistence is inverted: confounds are more persistent than true ET (AUC 0.126)."
        ),
        "02_growth_metric_method_trajectories.png": (
            "Per-method trajectories across baseline, spatial prior, soft shape proxy, and cubical persistence on the locked Stage-4 axes. Curves are mostly flat or degenerate: false-positive reductions often cost detection, and detection does not materially improve."
        ),
        "03_best_vs_worst_variational_spline_vs_gmm2d.png": (
            "Best detector (variational_spline) versus worst baseline method (gmm_2d), with bootstrap 95% intervals from existing case-level results. The best method has higher large-lesion Dice and lower surface ASD, while gmm_2d carries much larger absent-case FP burden."
        ),
        "04_per_stratum_lesionwise_and_surface_asd.png": (
            "Per-stratum performance for the five core methods. Absent cases have no lesion-wise Dice or surface ASD because GT ET is absent; they must be scored by false-positive burden. Present-case detection improves mainly in the large-tumor stratum."
        ),
        "05_absent_case_flood_and_fp_volume.png": (
            "Absent-case hallucination result on 33 ET-absent cohort cases. Every method predicts nonempty ET, and 94-100% of absent cases exceed the 10,000-voxel flood threshold, although median FP volume differs by method."
        ),
        "06_poisson_surface_renders_good_vs_irreducible.png": (
            "Poisson surface render panel. Case 02306 shows a good large-tumor contrast case; 00533 and 02078 show irreducible non-tumor enhancement confounds reconstructing as coherent but spatially wrong surfaces."
        ),
        "07_surface_vs_dice_large_stratum.png": (
            "Large-stratum lesion-wise Dice versus surface ASD. The negative Spearman relationship shows that the surface metric confirms rather than rescues the voxel metric: methods with low lesion-wise Dice also have worse surface distance."
        ),
    }
    path = OUT / "CAPTIONS.md"
    lines = ["# Figure Captions", ""]
    for p in paths:
        lines.append(f"## {p.name}")
        lines.append("")
        lines.append(captions[p.name])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_zip() -> Path:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                zf.write(path, arcname=f"figures/{path.name}")
    return ZIP_PATH


def main() -> None:
    if OUT.exists():
        for old in OUT.glob("*.png"):
            old.unlink()
        cap = OUT / "CAPTIONS.md"
        if cap.exists():
            cap.unlink()
    else:
        OUT.mkdir(parents=True, exist_ok=True)

    paths = [
        figure_impossibility_chain(),
        figure_growth_metric(),
        figure_best_worst(),
        figure_per_stratum(),
        figure_absent_flood(),
        figure_surface_renders(),
        figure_surface_vs_dice(),
    ]
    captions = write_captions(paths)
    zpath = make_zip()
    print("Generated figures:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")
    print(f"  {captions.relative_to(ROOT)}")
    print(f"Zip: {zpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
