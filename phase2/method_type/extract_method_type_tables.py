"""Assemble method-type and per-parameter behavior tables.

Reads existing Phase 1, Stage 4, P1, P2a, and P2b CSVs. Outputs summarize how
construction type relates to false-positive burden and detection behavior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase2" / "method_type"

METHOD_TYPES = {
    "otsu_T1c": ("intensity/statistical", "Otsu threshold over T1c intensities"),
    "gmm_T1c": ("intensity/statistical", "Gaussian mixture over T1c intensities"),
    "gmm_2d": ("intensity/statistical", "Gaussian mixture over 2D intensity feature space"),
    "sustraccion": ("intensity/statistical", "T1c-T1n enhancement subtraction/intensity contrast"),
    "variational_spline": ("deformable/region-based", "Chan-Vese evolving contour / region energy"),
    "level_set": ("deformable/region-based", "level-set PDE"),
    "bspline": ("deformable/control-point", "B-spline/control point enclosure"),
    "spline": ("deformable/control-point", "spline/control point enclosure"),
    "fast_marching": ("front-propagation", "fast marching front propagation"),
    "rango_doble": ("intensity/statistical", "double intensity range rule"),
}

CORE_METHODS = ["otsu_T1c", "gmm_T1c", "gmm_2d", "sustraccion", "variational_spline"]
ALL_METHODS = [
    "otsu_T1c", "gmm_T1c", "gmm_2d", "sustraccion", "rango_doble",
    "variational_spline", "level_set", "bspline", "spline", "fast_marching",
]

SOURCES = {
    "oracle_leave_one_out": "analysis/oracle_leave_one_out.csv",
    "keep_cut": "analysis/keep_cut_updated.csv",
    "stage4_present": "analysis/stage4_metrics/stage4_present_by_vol_bin.csv",
    "stage4_absent": "analysis/stage4_metrics/stage4_absent_fp_summary.csv",
    "stage4_ranking": "analysis/stage4_metrics/stage4_method_ranking_detection_vs_fp.csv",
    "p1_ops": "phase2/p1_spatial_atlas/tradeoff_analysis/p1_fixed_detection_cost_operating_points.csv",
    "p1_summary": "phase2/p1_spatial_atlas/p1_spatial_sweep_summary.csv",
    "p2a_key": "phase2/p2_shape_prior/p2_shape_key_comparison_table.csv",
    "p2b_ops": "phase2/p2_soft_shape_sweep/p2_soft_shape_operating_points.csv",
    "p2b_summary": "phase2/p2_soft_shape_sweep/p2_soft_shape_sweep_summary.csv",
}


def read_csv(key: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / SOURCES[key])


def blank() -> float:
    return np.nan


def build_method_matrix() -> pd.DataFrame:
    oracle = read_csv("oracle_leave_one_out")
    keep = read_csv("keep_cut")
    stage4_present = read_csv("stage4_present")
    stage4_absent = read_csv("stage4_absent")
    ranking = read_csv("stage4_ranking")
    p1_ops = read_csv("p1_ops")
    p2a = read_csv("p2a_key")
    p2b = read_csv("p2b_ops")

    rows = []
    for method in ALL_METHODS:
        method_type, construction = METHOD_TYPES[method]
        row = {
            "method": method,
            "construction_type": method_type,
            "construction_note": construction,
            "is_core_stage4_method": method in CORE_METHODS,
        }

        k = keep[keep["method"].eq(method)]
        if len(k):
            kr = k.iloc[0]
            row.update({
                "phase1_keep_cut": kr["recommendation"],
                "phase1_mean_dice_original20": kr["mean_dice"],
                "phase1_oracle_delta": kr["delta_full_minus_without"],
                "phase1_sole_best_cases": kr["sole_best_cases"],
                "phase1_sole_above_075_cases": kr["sole_above_075_cases"],
                "phase1_source": SOURCES["keep_cut"],
            })
        else:
            o = oracle[oracle["method"].eq(method)]
            if len(o):
                orow = o.iloc[0]
                row.update({
                    "phase1_keep_cut": "",
                    "phase1_mean_dice_original20": blank(),
                    "phase1_oracle_delta": orow["delta_full_minus_without"],
                    "phase1_sole_best_cases": orow["sole_best_cases"],
                    "phase1_sole_above_075_cases": orow["sole_above_075_cases"],
                    "phase1_source": SOURCES["oracle_leave_one_out"],
                })

        for vol in ["small", "medium", "large"]:
            sub = stage4_present[stage4_present["metodo"].eq(method) & stage4_present["vol_bin"].eq(vol)]
            if len(sub):
                sr = sub.iloc[0]
                row[f"stage4_{vol}_n"] = sr["n"]
                row[f"stage4_{vol}_lesionwise_mean"] = sr["lesionwise_dice_mean"]
                row[f"stage4_{vol}_global_dice_mean"] = sr["global_dice_mean"]
                row[f"stage4_{vol}_detection_rate_mean"] = sr["lesion_detection_rate_mean"]
                row[f"stage4_{vol}_source"] = SOURCES["stage4_present"]
            else:
                row[f"stage4_{vol}_n"] = blank()
                row[f"stage4_{vol}_lesionwise_mean"] = blank()
                row[f"stage4_{vol}_global_dice_mean"] = blank()
                row[f"stage4_{vol}_detection_rate_mean"] = blank()
                row[f"stage4_{vol}_source"] = ""

        abs_sub = stage4_absent[stage4_absent["metodo"].eq(method)]
        if len(abs_sub):
            ar = abs_sub.iloc[0]
            row.update({
                "stage4_absent_n": ar["n_absent"],
                "stage4_absent_flood_rate": ar["flood_gt_10000_rate"],
                "stage4_absent_median_fp_vox": ar["fp_volume_median"],
                "stage4_absent_max_fp_vox": ar["fp_volume_max"],
                "stage4_absent_source": SOURCES["stage4_absent"],
            })
        else:
            row.update({
                "stage4_absent_n": blank(),
                "stage4_absent_flood_rate": blank(),
                "stage4_absent_median_fp_vox": blank(),
                "stage4_absent_max_fp_vox": blank(),
                "stage4_absent_source": "",
            })

        rank_sub = ranking[ranking["metodo"].eq(method)]
        if len(rank_sub):
            rr = rank_sub.iloc[0]
            row.update({
                "stage4_present_lesionwise_all_mean": rr["present_lesionwise_dice_mean"],
                "stage4_present_lesion_tp": rr["present_lesion_tp"],
                "stage4_present_lesion_fn": rr["present_lesion_fn"],
                "stage4_present_lesion_fp": rr["present_lesion_fp"],
                "stage4_rank_sum": rr["rank_sum"],
                "stage4_ranking_source": SOURCES["stage4_ranking"],
            })
        else:
            row.update({
                "stage4_present_lesionwise_all_mean": blank(),
                "stage4_present_lesion_tp": blank(),
                "stage4_present_lesion_fn": blank(),
                "stage4_present_lesion_fp": blank(),
                "stage4_rank_sum": blank(),
                "stage4_ranking_source": "",
            })

        p1 = p1_ops[p1_ops["method"].eq(method)]
        if len(p1):
            pr = p1.iloc[0]
            row.update({
                "p1_mode": pr["mode"],
                "p1_threshold": pr["chosen_threshold"],
                "p1_absent_flood_rate": pr["absent_flood_rate"],
                "p1_absent_median_fp_vox": pr["absent_median_fp_vox"],
                "p1_large_lesionwise": pr["large_lesionwise"],
                "p1_large_detection_ratio": pr["large_detection_ratio"],
                "p1_large_dominant_found_rate": pr["large_dominant_found_rate"],
                "p1_source": SOURCES["p1_ops"],
            })
        else:
            row.update({
                "p1_mode": "",
                "p1_threshold": blank(),
                "p1_absent_flood_rate": blank(),
                "p1_absent_median_fp_vox": blank(),
                "p1_large_lesionwise": blank(),
                "p1_large_detection_ratio": blank(),
                "p1_large_dominant_found_rate": blank(),
                "p1_source": "",
            })

        p2a_sub = p2a[p2a["method"].eq(method)]
        if len(p2a_sub):
            hr = p2a_sub.iloc[0]
            row.update({
                "p2a_absent_flood_rate": hr["absent_flood_rate"],
                "p2a_absent_median_fp_vox": hr["absent_median_fp_vox"],
                "p2a_delta_flood_vs_otsu": hr["delta_absent_flood_vs_otsu"],
                "p2a_large_lesionwise": hr["large_lesionwise_mean"],
                "p2a_large_detection_ratio": hr["large_retention_vs_detection_baseline"],
                "p2a_large_dominant_found_rate": hr["large_dominant_found_rate"],
                "p2a_preserves_large_90pct": hr["large_detection_preserved_90pct"],
                "p2a_source": SOURCES["p2a_key"],
            })
        else:
            row.update({
                "p2a_absent_flood_rate": blank(),
                "p2a_absent_median_fp_vox": blank(),
                "p2a_delta_flood_vs_otsu": blank(),
                "p2a_large_lesionwise": blank(),
                "p2a_large_detection_ratio": blank(),
                "p2a_large_dominant_found_rate": blank(),
                "p2a_preserves_large_90pct": "",
                "p2a_source": "",
            })

        p2b_method = p2b[
            p2b["method"].eq(method)
            & p2b["selection"].eq("lowest_fp_while_large_detection_ge_90pct")
        ]
        if not len(p2b_method):
            p2b_method = p2b[p2b["method"].eq(method)].head(1)
        if len(p2b_method):
            br = p2b_method.iloc[0]
            row.update({
                "p2b_selection": br["selection"],
                "p2b_threshold": br["shape_score_threshold"],
                "p2b_absent_flood_rate": br["absent_flood_rate"],
                "p2b_absent_median_fp_vox": br["absent_median_fp_vox"],
                "p2b_delta_flood_vs_otsu": br["delta_absent_flood_vs_otsu"],
                "p2b_delta_flood_ci_low": br["delta_absent_flood_ci_low"],
                "p2b_delta_flood_ci_high": br["delta_absent_flood_ci_high"],
                "p2b_delta_median_fp_vs_otsu": br["delta_absent_median_fp_vs_otsu"],
                "p2b_large_lesionwise": br["large_lesionwise_mean"],
                "p2b_large_detection_ratio": br["large_detection_ratio"],
                "p2b_large_dominant_found_rate": br["large_dominant_found_rate"],
                "p2b_clears_clean_bar": br["clears_clean_bar"],
                "p2b_source": SOURCES["p2b_ops"],
            })
        else:
            row.update({
                "p2b_selection": "",
                "p2b_threshold": blank(),
                "p2b_absent_flood_rate": blank(),
                "p2b_absent_median_fp_vox": blank(),
                "p2b_delta_flood_vs_otsu": blank(),
                "p2b_delta_flood_ci_low": blank(),
                "p2b_delta_flood_ci_high": blank(),
                "p2b_delta_median_fp_vs_otsu": blank(),
                "p2b_large_lesionwise": blank(),
                "p2b_large_detection_ratio": blank(),
                "p2b_large_dominant_found_rate": blank(),
                "p2b_clears_clean_bar": "",
                "p2b_source": "",
            })

        rows.append(row)

    out = pd.DataFrame(rows)
    out["type_order"] = out["construction_type"].map({
        "intensity/statistical": 0,
        "deformable/region-based": 1,
        "deformable/control-point": 2,
        "front-propagation": 3,
    }).fillna(99)
    out = out.sort_values(["type_order", "method"]).drop(columns=["type_order"])
    out.to_csv(OUT / "method_type_stage_matrix.csv", index=False)
    return out


def first_threshold(sub: pd.DataFrame, condition) -> float:
    hits = sub[condition(sub)].sort_values("shape_score_threshold")
    if hits.empty:
        return np.nan
    return float(hits.iloc[0]["shape_score_threshold"])


def build_per_parameter_behavior() -> pd.DataFrame:
    p1 = read_csv("p1_summary")
    p2b = read_csv("p2b_summary")
    rows = []

    p1_method_modes = p1[p1["method"].isin(CORE_METHODS) | p1["method"].eq("atlas_map")]
    for _, r in p1_method_modes.iterrows():
        method = r["method"]
        method_type = "atlas/map" if method == "atlas_map" else METHOD_TYPES[method][0]
        rows.append({
            "prior_stage": "P1_spatial",
            "method": method,
            "construction_type": method_type,
            "mode": r["mode"],
            "threshold": r["threshold"],
            "absent_flood_rate": r["absent_flood_rate"],
            "absent_median_fp_vox": r["absent_median_fp_vox"],
            "large_lesionwise": r["large_lesionwise_mean"],
            "large_detection_ratio": r["large_lesionwise_mean"] / 0.1762380286972449,
            "keeps_large_detection_90pct": r["large_lesionwise_mean"] >= 0.9 * 0.1762380286972449,
            "source": SOURCES["p1_summary"],
        })

    for _, r in p2b.iterrows():
        method = r["method"]
        rows.append({
            "prior_stage": "P2b_soft_shape",
            "method": method,
            "construction_type": METHOD_TYPES[method][0],
            "mode": "shape_score_threshold",
            "threshold": r["shape_score_threshold"],
            "absent_flood_rate": r["absent_flood_rate"],
            "absent_median_fp_vox": r["absent_median_fp_vox"],
            "large_lesionwise": r["large_lesionwise_mean"],
            "large_detection_ratio": r["large_detection_ratio"],
            "keeps_large_detection_90pct": r["keeps_large_detection_90pct"],
            "source": SOURCES["p2b_summary"],
        })

    behavior = pd.DataFrame(rows)
    behavior.to_csv(OUT / "per_parameter_behavior.csv", index=False)
    return behavior


def usable_window_table() -> pd.DataFrame:
    p2b = read_csv("p2b_summary")
    rows = []
    baseline = read_csv("stage4_absent")
    for method in CORE_METHODS:
        sub = p2b[p2b["method"].eq(method)].sort_values("shape_score_threshold").copy()
        base_abs = baseline[baseline["metodo"].eq(method)].iloc[0]
        first_flood_below_own = first_threshold(
            sub, lambda d: d["absent_flood_rate"] < float(base_abs["flood_gt_10000_rate"]))
        first_flood_below_otsu = first_threshold(
            sub, lambda d: d["absent_flood_rate"] < 0.9393939393939394)
        first_detection_below_90 = first_threshold(
            sub, lambda d: d["large_detection_ratio"] < 0.90)
        viable = sub[
            (sub["absent_flood_rate"] < 0.9393939393939394)
            & (sub["absent_median_fp_vox"] < 13935.0)
            & (sub["large_detection_ratio"] >= 0.90)
        ].copy()
        loose_viable = sub[
            (sub["absent_flood_rate"] < float(base_abs["flood_gt_10000_rate"]))
            & (sub["large_detection_ratio"] >= 0.90)
        ].copy()
        rows.append({
            "method": method,
            "construction_type": METHOD_TYPES[method][0],
            "baseline_own_flood_rate": base_abs["flood_gt_10000_rate"],
            "baseline_own_median_fp_vox": base_abs["fp_volume_median"],
            "first_threshold_flood_below_own_baseline": first_flood_below_own,
            "first_threshold_flood_below_otsu_baseline": first_flood_below_otsu,
            "first_threshold_large_detection_below_90pct": first_detection_below_90,
            "has_clean_usable_window_vs_otsu": bool(len(viable)),
            "has_loose_usable_window_vs_own_baseline": bool(len(loose_viable)),
            "clean_window_threshold_min": viable["shape_score_threshold"].min() if len(viable) else np.nan,
            "clean_window_threshold_max": viable["shape_score_threshold"].max() if len(viable) else np.nan,
            "loose_window_threshold_min": loose_viable["shape_score_threshold"].min() if len(loose_viable) else np.nan,
            "loose_window_threshold_max": loose_viable["shape_score_threshold"].max() if len(loose_viable) else np.nan,
            "source": SOURCES["p2b_summary"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "per_parameter_usable_windows.csv", index=False)
    return out


def fmt(x, digits=3):
    if pd.isna(x):
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{x:.{digits}f}"
    return str(x)


def write_method_type_findings(matrix: pd.DataFrame) -> None:
    core = matrix[matrix["is_core_stage4_method"]].copy()
    intensity = core[core["construction_type"].eq("intensity/statistical")]
    deform = core[core["construction_type"].eq("deformable/region-based")]
    text = f"""# Method Type x Stage Findings

Source table: `phase2/method_type/method_type_stage_matrix.csv`

## Construction-Type Classification

- Intensity/statistical: `otsu_T1c`, `gmm_T1c`, `gmm_2d`, `sustraccion`, plus cut method `rango_doble`.
- Deformable/region-based: `variational_spline`, plus cut method `level_set`.
- Deformable/control-point: cut methods `bspline`, `spline`.
- Front-propagation: cut method `fast_marching`.

## Does Construction Type Predict Behavior?

Partly, but not cleanly enough to use as a deterministic rule.

### Stage 4 Baseline

The intensity/statistical core methods do flood strongly on ET-absent cases:

- `otsu_T1c`: flood `{fmt(float(core[core.method.eq('otsu_T1c')]['stage4_absent_flood_rate'].iloc[0]))}`, median FP `{fmt(float(core[core.method.eq('otsu_T1c')]['stage4_absent_median_fp_vox'].iloc[0]), 0)}` vox.
- `gmm_T1c`: flood `{fmt(float(core[core.method.eq('gmm_T1c')]['stage4_absent_flood_rate'].iloc[0]))}`, median FP `{fmt(float(core[core.method.eq('gmm_T1c')]['stage4_absent_median_fp_vox'].iloc[0]), 0)}` vox.
- `gmm_2d`: flood `{fmt(float(core[core.method.eq('gmm_2d')]['stage4_absent_flood_rate'].iloc[0]))}`, median FP `{fmt(float(core[core.method.eq('gmm_2d')]['stage4_absent_median_fp_vox'].iloc[0]), 0)}` vox.
- `sustraccion`: flood `{fmt(float(core[core.method.eq('sustraccion')]['stage4_absent_flood_rate'].iloc[0]))}`, median FP `{fmt(float(core[core.method.eq('sustraccion')]['stage4_absent_median_fp_vox'].iloc[0]), 0)}` vox.

The deformable/region-based `variational_spline` is more restrained by maximum FP than the worst intensity methods, but it still floods:

- `variational_spline`: flood `{fmt(float(deform['stage4_absent_flood_rate'].iloc[0]))}`, median FP `{fmt(float(deform['stage4_absent_median_fp_vox'].iloc[0]), 0)}` vox, max FP `{fmt(float(deform['stage4_absent_max_fp_vox'].iloc[0]), 0)}` vox.

For present tumors, `variational_spline` has the best large-stratum lesion-wise score (`{fmt(float(deform['stage4_large_lesionwise_mean'].iloc[0]))}`), consistent with the idea that the region-based method better preserves a main mass. But it does not solve small or multifocal detection.

### Phase 1 Oracle/Redundancy

Construction type alone did not decide keep/cut. The five core methods all had positive leave-one-out oracle deltas, while the legacy methods had zero delta. This supports pruning by measured unique value, not by type label alone.

### P1 Spatial Prior

P1's useful operating point was reported only for `variational_spline` post-filter. It reduced flood to `0.818` at `92.2%` large-detection retention but worsened median FP relative to Otsu. This does not establish a general type effect.

### P2a Hard Shape Filter

P2a shows the clearest type interaction:

- `variational_spline` retained `{fmt(float(deform['p2a_large_detection_ratio'].iloc[0]) * 100, 1)}%` of large detection under hard shape filtering.
- `otsu_T1c` retained only `{fmt(float(core[core.method.eq('otsu_T1c')]['p2a_large_detection_ratio'].iloc[0]) * 100, 1)}%`.
- `gmm_T1c` retained `{fmt(float(core[core.method.eq('gmm_T1c')]['p2a_large_detection_ratio'].iloc[0]) * 100, 1)}%`.

This supports the hypothesis that deformable predictions are more shape-compatible than raw intensity-threshold predictions. However, P2a still failed the 90% detection-preservation gate.

### P2b Soft Shape Prior

P2b complicates the type story:

- The clean-bar exploratory point is `otsu_T1c` at threshold `0.010`, flood `0.788`, median FP `13,772`, large retention `93.6%`.
- `variational_spline` has a stronger flood reduction (`0.727`) and retains `92.6%` large detection, but median FP is `14,118`, slightly worse than Otsu's baseline.

Interpretation: type predicts baseline morphology and hard-filter survivability, but the best soft operating point depends on the starting method's FP distribution. The strongest paper-safe statement is:

> Intensity/statistical methods tend to detect by broad intensity capture and flood ET-absent cases; the deformable method is more compact and better aligned with shape filtering, but it still floods and misses satellites. Construction type explains tendencies, not outcomes.
"""
    (OUT / "method_type_findings.md").write_text(text, encoding="utf-8")


def write_parameter_findings(windows: pd.DataFrame) -> None:
    clean = windows[windows["has_clean_usable_window_vs_otsu"]]
    loose = windows[windows["has_loose_usable_window_vs_own_baseline"]]
    text = f"""# Per-Parameter Behavior Findings

Source tables:

- `phase2/method_type/per_parameter_behavior.csv`
- `phase2/method_type/per_parameter_usable_windows.csv`

## P1 Spatial Sweep

The P1 table records the full threshold curves from `p1_spatial_sweep_summary.csv` for pre-filter, post-filter, and MAP modes. The headline read remains:

- MAP can drive flood to zero only by collapsing detection.
- Pre-filter has a sharp flood/detection cliff.
- Post-filter on `variational_spline` has the only modest P1 operating point: flood `0.818`, median FP `18,518`, large retention `92.2%`.

## P2b Soft Shape Sweep

Clean usable window definition:

- absent flood below Otsu baseline `0.939`
- median absent FP below Otsu baseline `13,935`
- large detection retention at least `90%`

Methods with a clean usable window:

{', '.join(clean['method'].tolist()) if len(clean) else 'none'}

Loose usable window definition:

- absent flood below the method's own baseline
- large detection retention at least `90%`

Methods with a loose usable window:

{', '.join(loose['method'].tolist()) if len(loose) else 'none'}

## Window Details

| method | type | clean window | loose window | first flood drop vs own | first large drop below 90% |
|---|---|---|---:|---:|
"""
    for _, r in windows.iterrows():
        clean_win = "yes" if r["has_clean_usable_window_vs_otsu"] else "no"
        if r["has_clean_usable_window_vs_otsu"]:
            clean_win += f" ({fmt(r['clean_window_threshold_min'])}-{fmt(r['clean_window_threshold_max'])})"
        loose_win = "yes" if r["has_loose_usable_window_vs_own_baseline"] else "no"
        if r["has_loose_usable_window_vs_own_baseline"]:
            loose_win += f" ({fmt(r['loose_window_threshold_min'])}-{fmt(r['loose_window_threshold_max'])})"
        text += (
            f"| `{r['method']}` | {r['construction_type']} | {clean_win} | {loose_win} | "
            f"{fmt(r['first_threshold_flood_below_own_baseline'])} | "
            f"{fmt(r['first_threshold_large_detection_below_90pct'])} |\n"
        )

    text += """
## Type Interaction

The hypothesis was that the deformable method would have the widest usable shape-prior window because its true-tumor predictions are already compact.

The data only partly support this:

- `variational_spline` does have a loose usable window and the best flood reduction while preserving large detection: threshold `0.020`, flood `0.727`, large retention `92.6%`.
- But it misses the clean window because median FP remains slightly worse than Otsu baseline (`14,118` vs `13,935`).
- `otsu_T1c`, an intensity/statistical method, is the only method satisfying the stricter clean window, because its baseline median FP is already low and the soft shape threshold trims just enough FP while preserving large detection.
- `gmm_T1c` preserves detection at threshold `0`, but has no useful FP reduction window before detection degrades.
- `gmm_2d` and `sustraccion` have no meaningful clean shape-prior window.

Conclusion:

> Shape prior response is not determined by construction type alone. Deformable masks are more shape-compatible under aggressive filtering, but the clinically useful soft operating point is controlled by the joint distribution of baseline FP volume and true-tumor component scores.
"""
    (OUT / "per_parameter_findings.md").write_text(text, encoding="utf-8")


def main() -> None:
    matrix = build_method_matrix()
    behavior = build_per_parameter_behavior()
    windows = usable_window_table()
    write_method_type_findings(matrix)
    write_parameter_findings(windows)
    print(f"Wrote {OUT / 'method_type_stage_matrix.csv'} rows={len(matrix)}")
    print(f"Wrote {OUT / 'method_type_findings.md'}")
    print(f"Wrote {OUT / 'per_parameter_behavior.csv'} rows={len(behavior)}")
    print(f"Wrote {OUT / 'per_parameter_usable_windows.csv'} rows={len(windows)}")
    print(f"Wrote {OUT / 'per_parameter_findings.md'}")


if __name__ == "__main__":
    main()
