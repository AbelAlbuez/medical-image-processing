"""Summarize P1 spatial-prior flood/detection tradeoff curves.

Reads existing P1 sweep CSVs only. No segmentation or atlas recomputation is
performed here; this script selects/report operating points for interpretation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


P1 = ROOT / "phase2" / "p1_spatial_atlas"
ATLASES = P1 / "atlases"
OUT = P1 / "tradeoff_analysis"
IRREDUCIBLE = ROOT / "analysis" / "stage3d_irreducible_mechanism.csv"
BASELINE_TARGETS = ROOT / "phase2" / "baseline_targets_per_axis.csv"
BASELINE_CASE = ROOT / "phase2" / "p0_metric_recomputed_case_metrics.csv"

LARGE_BASELINE = 0.1762380286972449
LARGE_90 = 0.90 * LARGE_BASELINE


def parse_centroid(text: str) -> tuple[int, int, int]:
    vals = [float(x) for x in str(text).split(";")]
    return tuple(int(round(v)) for v in vals)


def dominance_baseline() -> pd.DataFrame:
    # Use Stage-4 final-read component diagnostics for interpretable dominant-found rates.
    detail = pd.read_csv(ROOT / "analysis" / "stage4_final_read" / "stage4_final_read_component_detail.csv")
    rows = []
    for method, sub in detail.groupby("metodo"):
        present = sub
        large = sub[sub["vol_bin"].eq("large")]
        rows.append({
            "mode": "baseline",
            "method": method,
            "threshold": np.nan,
            "present_dominant_found_rate": float(present["dominant_found"].mean()),
            "large_dominant_found_rate": float(large["dominant_found"].mean()),
        })
    return pd.DataFrame(rows)


def tradeoff_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(P1 / "p1_spatial_sweep_summary.csv")
    cases = pd.read_csv(P1 / "p1_spatial_sweep_case_metrics.csv")

    # Dominant-lesion-found proxy from locked metrics: at least one TP on present cases.
    dom = cases[cases["vol_bin"].isin(["small", "medium", "large"])].copy()
    dom["dominant_found_proxy"] = dom["lesion_tp"].fillna(0) > 0
    dom_summary = dom.groupby(["mode", "method", "threshold"]).agg(
        present_dominant_found_rate=("dominant_found_proxy", "mean"),
        large_dominant_found_rate=("dominant_found_proxy", lambda s: float(s[dom.loc[s.index, "vol_bin"].eq("large")].mean())),
    ).reset_index()

    curves = summary.merge(dom_summary, on=["mode", "method", "threshold"], how="left")
    curves["large_detection_ratio_vs_baseline"] = curves["large_lesionwise_mean"] / LARGE_BASELINE
    curves["large_detection_drop"] = 1.0 - curves["large_detection_ratio_vs_baseline"]
    curves["keeps_large_detection_90pct"] = curves["large_lesionwise_mean"] >= LARGE_90
    curves.to_csv(OUT / "p1_tradeoff_all_configs.csv", index=False)

    # Representative curves: MAP has atlas_map; filters use variational_spline as the
    # restrained Stage-4 method that P1 was meant to help.
    reps = [
        ("map_atlas_x_enhancement", "atlas_map"),
        ("post_filter", "variational_spline"),
        ("pre_filter", "variational_spline"),
    ]
    rep_rows = []
    for mode, method in reps:
        sub = curves[curves["mode"].eq(mode) & curves["method"].eq(method)].copy()
        sub = sub.sort_values("threshold")
        sub["representative_curve"] = True
        rep_rows.append(sub)
    rep = pd.concat(rep_rows, ignore_index=True)
    rep.to_csv(OUT / "p1_tradeoff_representative_curves.csv", index=False)

    return curves, rep


def operating_points(rep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, method), sub in rep.groupby(["mode", "method"]):
        sub = sub.sort_values("threshold")
        ok = sub[sub["keeps_large_detection_90pct"]].copy()
        if ok.empty:
            chosen = sub.iloc[(sub["large_lesionwise_mean"] - LARGE_90).abs().argsort().iloc[0]]
            selection = "closest_to_90pct_no_config_preserves_90pct"
        else:
            chosen = ok.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
            selection = "lowest_fp_while_large_detection_ge_90pct"

        first_drop = sub[sub["large_detection_drop"] > 0.10].sort_values("threshold").head(1)
        if first_drop.empty:
            first_drop_threshold = np.nan
            first_drop_flood = np.nan
        else:
            first_drop_threshold = float(first_drop.iloc[0]["threshold"])
            first_drop_flood = float(first_drop.iloc[0]["absent_flood_rate"])

        rows.append({
            "mode": mode,
            "method": method,
            "selection": selection,
            "chosen_threshold": float(chosen["threshold"]),
            "large_baseline": LARGE_BASELINE,
            "large_90pct_target": LARGE_90,
            "large_lesionwise": float(chosen["large_lesionwise_mean"]),
            "large_detection_ratio": float(chosen["large_detection_ratio_vs_baseline"]),
            "absent_flood_rate": float(chosen["absent_flood_rate"]),
            "absent_median_fp_vox": float(chosen["absent_median_fp_vox"]),
            "present_dominant_found_rate": float(chosen["present_dominant_found_rate"]),
            "large_dominant_found_rate": float(chosen["large_dominant_found_rate"]),
            "first_threshold_large_drop_gt_10pct": first_drop_threshold,
            "flood_at_first_large_drop_gt_10pct": first_drop_flood,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p1_fixed_detection_cost_operating_points.csv", index=False)
    return out


def plot_curves(rep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    labels = {
        "map_atlas_x_enhancement": "MAP atlas*enh",
        "post_filter": "post-filter varspline",
        "pre_filter": "pre-filter varspline",
    }
    for (mode, method), sub in rep.groupby(["mode", "method"]):
        sub = sub.sort_values("threshold")
        ax.plot(sub["absent_flood_rate"], sub["large_lesionwise_mean"],
                marker="o", label=labels.get(mode, f"{mode}:{method}"))
        for _, row in sub.iterrows():
            ax.annotate(f"{row['threshold']:.3g}",
                        (row["absent_flood_rate"], row["large_lesionwise_mean"]),
                        fontsize=7, alpha=0.75)
    ax.axhline(LARGE_BASELINE, color="black", linestyle="--", linewidth=1, label="large baseline")
    ax.axhline(LARGE_90, color="gray", linestyle=":", linewidth=1, label="90% baseline")
    ax.axvline(0.9393939393939394, color="red", linestyle="--", linewidth=1, label="otsu flood")
    ax.set_xlabel("Absent-case flood rate")
    ax.set_ylabel("Large-stratum FP-aware lesion-wise Dice")
    ax.set_title("P1 spatial-prior FP vs large-detection tradeoff")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "p1_tradeoff_curves.png", dpi=180)
    plt.close(fig)


def seed_percentiles() -> pd.DataFrame:
    irr = pd.read_csv(IRREDUCIBLE)
    atlas_arrays = {}
    for path in sorted(ATLASES.glob("et_occurrence_atlas_holdout_fold*.nii.gz")):
        fold = int(path.stem.split("fold")[-1].split(".")[0])
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        atlas_arrays[fold] = arr

    rows = []
    for _, row in irr.iterrows():
        z, y, x = parse_centroid(row["gmm_centroid_zyx"])
        for fold, arr in atlas_arrays.items():
            value = float(arr[z, y, x])
            brain = arr[arr > 0]
            percentile_nonzero = float((brain < value).mean() * 100.0) if brain.size else np.nan
            percentile_all = float((arr < value).mean() * 100.0)
            rows.append({
                "case_id": row["case_id"],
                "fold": fold,
                "gmm_centroid_zyx": row["gmm_centroid_zyx"],
                "atlas_value": value,
                "atlas_max": float(arr.max()),
                "value_over_max": value / float(arr.max()) if arr.max() > 0 else np.nan,
                "percentile_among_nonzero_atlas_voxels": percentile_nonzero,
                "percentile_among_all_grid_voxels": percentile_all,
                "fraction_nonzero_voxels_below_seed": percentile_nonzero / 100.0,
            })
    out = pd.DataFrame(rows)
    summary = out.groupby("case_id").agg(
        atlas_value_mean=("atlas_value", "mean"),
        value_over_max_mean=("value_over_max", "mean"),
        percentile_nonzero_mean=("percentile_among_nonzero_atlas_voxels", "mean"),
        percentile_nonzero_min=("percentile_among_nonzero_atlas_voxels", "min"),
        percentile_nonzero_max=("percentile_among_nonzero_atlas_voxels", "max"),
        percentile_all_mean=("percentile_among_all_grid_voxels", "mean"),
    ).reset_index()
    out.to_csv(OUT / "p1_seed_percentile_detail.csv", index=False)
    summary.to_csv(OUT / "p1_seed_percentile_summary.csv", index=False)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves, rep = tradeoff_curves()
    ops = operating_points(rep)
    plot_curves(rep)
    seed_pct = seed_percentiles()

    print("\nREPRESENTATIVE TRADEOFF CURVES")
    cols = [
        "mode", "method", "threshold", "absent_flood_rate", "absent_median_fp_vox",
        "large_lesionwise_mean", "large_detection_ratio_vs_baseline",
        "present_dominant_found_rate", "large_dominant_found_rate",
    ]
    print(rep[cols].sort_values(["mode", "method", "threshold"]).to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nFIXED DETECTION COST OPERATING POINTS")
    print(ops.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSEED PERCENTILES")
    print(seed_pct.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nPlot: {OUT / 'p1_tradeoff_curves.png'}")


if __name__ == "__main__":
    main()
