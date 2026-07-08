"""Assemble comparable locked-metric growth tables across baseline/P1/P2b/P3.

Reads existing result CSVs only. The output tracks each core method on the same
Stage-4 axes: absent flood, absent median FP, and large-stratum lesion-wise Dice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase2" / "growth_metric"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
PHASES = ["baseline", "p1_spatial", "p2b_soft_shape", "p3_cubical_persistence"]
FLOOD_USABLE_BAR = 0.80
LARGE_RETENTION_BAR = 0.90

STAGE4_ABSENT = ROOT / "analysis" / "stage4_metrics" / "stage4_absent_fp_summary.csv"
STAGE4_PRESENT = ROOT / "analysis" / "stage4_metrics" / "stage4_present_by_vol_bin.csv"
P1_SWEEP = ROOT / "phase2" / "p1_spatial_atlas" / "p1_spatial_sweep_summary.csv"
P1_FIXED = ROOT / "phase2" / "p1_spatial_atlas" / "tradeoff_analysis" / "p1_fixed_detection_cost_operating_points.csv"
P2B_OPS = ROOT / "phase2" / "p2_soft_shape_sweep" / "p2_soft_shape_operating_points.csv"
P3_KEY = ROOT / "phase2" / "p3_cubical_persistence" / "p3_key_comparison_vs_baseline.csv"
USABLE = ROOT / "phase2" / "per_parameter_usable_windows.csv"


def baseline_rows() -> pd.DataFrame:
    absent = pd.read_csv(STAGE4_ABSENT)
    present = pd.read_csv(STAGE4_PRESENT)
    large = present[present["vol_bin"].eq("large")][["metodo", "lesionwise_dice_mean"]]
    df = absent.merge(large, on="metodo", how="left")
    return pd.DataFrame({
        "method": df["metodo"],
        "phase": "baseline",
        "phase_source": str(STAGE4_ABSENT.relative_to(ROOT)) + " + " + str(STAGE4_PRESENT.relative_to(ROOT)),
        "operating_point": "stage4_baseline_core5",
        "absent_flood_rate": df["flood_gt_10000_rate"],
        "absent_median_fp_vox": df["fp_volume_median"],
        "large_lesionwise_dice": df["lesionwise_dice_mean"],
        "valid_operating_point": True,
        "selection_note": "raw baseline prediction",
    })


def choose_p1_rows(base: pd.DataFrame) -> pd.DataFrame:
    sweep = pd.read_csv(P1_SWEEP)
    fixed = pd.read_csv(P1_FIXED)
    rows: List[Dict[str, object]] = []
    base_large = base.set_index("method")["large_lesionwise_dice"].to_dict()
    for method in METHODS:
        sub = sweep[sweep["method"].eq(method)].copy()
        if sub.empty:
            rows.append(empty_row(method, "p1_spatial", str(P1_FIXED.relative_to(ROOT)), "no method-level P1 row"))
            continue
        target = LARGE_RETENTION_BAR * float(base_large[method])
        feasible = sub[sub["large_lesionwise_mean"] >= target].copy()
        if feasible.empty:
            chosen = sub.iloc[(sub["large_lesionwise_mean"] - target).abs().argsort().iloc[0]]
            valid = False
            note = "closest_to_90pct_large_detection_no_valid_gate"
        else:
            chosen = feasible.sort_values(
                ["absent_flood_rate", "absent_median_fp_vox", "threshold"],
                ascending=[True, True, True],
            ).iloc[0]
            valid = True
            note = "lowest_fp_with_method_large_detection_ge_90pct"
        rows.append({
            "method": method,
            "phase": "p1_spatial",
            "phase_source": str(P1_SWEEP.relative_to(ROOT)),
            "operating_point": f"{chosen['mode']}@{float(chosen['threshold']):g}",
            "absent_flood_rate": float(chosen["absent_flood_rate"]),
            "absent_median_fp_vox": float(chosen["absent_median_fp_vox"]),
            "large_lesionwise_dice": float(chosen["large_lesionwise_mean"]),
            "valid_operating_point": bool(valid),
            "selection_note": note,
        })
    # Preserve the original headline P1 file as a separate audit trail.
    fixed.to_csv(OUT / "growth_p1_fixed_source_copy.csv", index=False)
    return pd.DataFrame(rows)


def choose_p2b_rows() -> pd.DataFrame:
    ops = pd.read_csv(P2B_OPS)
    rows = []
    for method in METHODS:
        sub = ops[ops["method"].eq(method)].copy()
        if sub.empty:
            rows.append(empty_row(method, "p2b_soft_shape", str(P2B_OPS.relative_to(ROOT)), "no P2b operating point"))
            continue
        preferred = sub[sub["selection"].eq("lowest_fp_while_large_detection_ge_90pct")]
        if preferred.empty:
            preferred = sub[sub["selection"].eq("closest_to_90pct_no_config_preserves_90pct")]
        if preferred.empty:
            preferred = sub.head(1)
        row = preferred.iloc[0]
        rows.append({
            "method": method,
            "phase": "p2b_soft_shape",
            "phase_source": str(P2B_OPS.relative_to(ROOT)),
            "operating_point": f"shape_score_threshold@{float(row['shape_score_threshold']):g}",
            "absent_flood_rate": float(row["absent_flood_rate"]),
            "absent_median_fp_vox": float(row["absent_median_fp_vox"]),
            "large_lesionwise_dice": float(row["large_lesionwise_mean"]),
            "valid_operating_point": bool(row["keeps_large_detection_90pct"]),
            "selection_note": str(row["selection"]),
            "delta_flood_ci_low": row.get("delta_absent_flood_ci_low", np.nan),
            "delta_flood_ci_high": row.get("delta_absent_flood_ci_high", np.nan),
            "clears_clean_bar": bool(row.get("clears_clean_bar", False)),
        })
    return pd.DataFrame(rows)


def p3_rows() -> pd.DataFrame:
    p3 = pd.read_csv(P3_KEY)
    rows = []
    for method in METHODS:
        sub = p3[p3["method"].eq(method)]
        if sub.empty:
            rows.append(empty_row(method, "p3_cubical_persistence", str(P3_KEY.relative_to(ROOT)), "no P3 operating point"))
            continue
        row = sub.iloc[0]
        rows.append({
            "method": method,
            "phase": "p3_cubical_persistence",
            "phase_source": str(P3_KEY.relative_to(ROOT)),
            "operating_point": "train_fold_top_k_persistent_components",
            "absent_flood_rate": float(row["absent_flood_rate"]),
            "absent_median_fp_vox": float(row["absent_median_fp_vox"]),
            "large_lesionwise_dice": float(row["large_lesionwise_mean"]),
            "valid_operating_point": bool(row["large_detection_preserved_90pct"]),
            "selection_note": "fold-learned count/rank prior",
            "delta_flood_ci_low": row.get("delta_absent_flood_ci_low", np.nan),
            "delta_flood_ci_high": row.get("delta_absent_flood_ci_high", np.nan),
        })
    return pd.DataFrame(rows)


def empty_row(method: str, phase: str, source: str, note: str) -> Dict[str, object]:
    return {
        "method": method,
        "phase": phase,
        "phase_source": source,
        "operating_point": "",
        "absent_flood_rate": np.nan,
        "absent_median_fp_vox": np.nan,
        "large_lesionwise_dice": np.nan,
        "valid_operating_point": False,
        "selection_note": note,
    }


def wide_growth_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub = long[long["method"].eq(method)].set_index("phase")
        row: Dict[str, object] = {"method": method}
        for phase in PHASES:
            if phase in sub.index:
                r = sub.loc[phase]
                for metric in ["absent_flood_rate", "absent_median_fp_vox", "large_lesionwise_dice"]:
                    row[f"{phase}_{metric}"] = r.get(metric, np.nan)
                row[f"{phase}_operating_point"] = r.get("operating_point", "")
                row[f"{phase}_valid_operating_point"] = r.get("valid_operating_point", False)
            else:
                for metric in ["absent_flood_rate", "absent_median_fp_vox", "large_lesionwise_dice"]:
                    row[f"{phase}_{metric}"] = np.nan
                row[f"{phase}_operating_point"] = ""
                row[f"{phase}_valid_operating_point"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def best_worst(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, sub in long.dropna(subset=["absent_flood_rate", "large_lesionwise_dice"]).groupby("phase"):
        fp_sorted = sub.sort_values(["absent_flood_rate", "absent_median_fp_vox"], ascending=[True, True])
        det_sorted = sub.sort_values("large_lesionwise_dice", ascending=False)
        for axis, kind, row in [
            ("FP-restraint", "best", fp_sorted.iloc[0]),
            ("FP-restraint", "worst", fp_sorted.iloc[-1]),
            ("Detection", "best", det_sorted.iloc[0]),
            ("Detection", "worst", det_sorted.iloc[-1]),
        ]:
            rows.append({
                "phase": phase,
                "axis": axis,
                "best_or_worst": kind,
                "method": row["method"],
                "absent_flood_rate": row["absent_flood_rate"],
                "absent_median_fp_vox": row["absent_median_fp_vox"],
                "large_lesionwise_dice": row["large_lesionwise_dice"],
                "operating_point": row["operating_point"],
            })
    out = pd.DataFrame(rows)
    phase_rank = {phase: i for i, phase in enumerate(PHASES)}
    out["phase_rank"] = out["phase"].map(phase_rank)
    return out.sort_values(["phase_rank", "axis", "best_or_worst"]).drop(columns=["phase_rank"])


def improvement_table(long: pd.DataFrame) -> pd.DataFrame:
    usable = pd.read_csv(USABLE).set_index("method")
    baseline_best_large = float(
        long[long["phase"].eq("baseline")]["large_lesionwise_dice"].max()
    )
    detection_floor_for_fp = LARGE_RETENTION_BAR * baseline_best_large
    rows = []
    for method in METHODS:
        sub = long[long["method"].eq(method)].copy()
        base = sub[sub["phase"].eq("baseline")].iloc[0]
        priors = sub[sub["phase"].ne("baseline") & sub["absent_flood_rate"].notna()].copy()
        if priors.empty:
            continue
        best_fp = priors.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
        best_det = priors.sort_values("large_lesionwise_dice", ascending=False).iloc[0]
        own = usable.loc[method] if method in usable.index else pd.Series(dtype=object)
        ci_low = best_fp.get("delta_flood_ci_low", np.nan)
        ci_high = best_fp.get("delta_flood_ci_high", np.nan)
        ci_excludes_zero = bool(np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_high < 0 or ci_low > 0))
        clean_window = bool(own.get("has_clean_usable_window_vs_otsu", False))
        loose_window = bool(own.get("has_loose_usable_window_vs_own_baseline", False))
        fp_improved = float(best_fp["absent_flood_rate"]) < float(base["absent_flood_rate"])
        fp_detection_preserved_vs_best = float(best_fp["large_lesionwise_dice"]) >= detection_floor_for_fp
        if ci_excludes_zero and clean_window and fp_detection_preserved_vs_best:
            fp_label = "REAL exploratory: CI excludes zero and clean usable window"
        elif fp_improved and not fp_detection_preserved_vs_best:
            fp_label = "DEGENERATE/MARGINAL: FP improves only by sacrificing usable large detection"
        elif fp_improved:
            fp_label = "SUGGESTIVE: FP improves but post-hoc/no clean usable window"
        else:
            fp_label = "NO FP improvement"
        det_crosses_best_gate = float(best_det["large_lesionwise_dice"]) >= detection_floor_for_fp
        det_label = (
            "SUGGESTIVE recovery: crosses 90% of best large detector, but not robustly established"
            if float(best_det["large_lesionwise_dice"]) > float(base["large_lesionwise_dice"]) + 0.01
            and det_crosses_best_gate else
            "MARGINAL numeric gain: still below 90% of best detector"
            if float(best_det["large_lesionwise_dice"]) > float(base["large_lesionwise_dice"]) + 0.01 else
            "NO meaningful detection improvement"
        )
        rows.append({
            "method": method,
            "baseline_flood": base["absent_flood_rate"],
            "best_prior_for_fp": best_fp["phase"],
            "best_prior_flood": best_fp["absent_flood_rate"],
            "delta_flood": best_fp["absent_flood_rate"] - base["absent_flood_rate"],
            "baseline_median_fp": base["absent_median_fp_vox"],
            "best_prior_median_fp": best_fp["absent_median_fp_vox"],
            "delta_median_fp": best_fp["absent_median_fp_vox"] - base["absent_median_fp_vox"],
            "best_fp_large_lesionwise": best_fp["large_lesionwise_dice"],
            "best_fp_preserves_90pct_of_best_detector": fp_detection_preserved_vs_best,
            "baseline_large_lesionwise": base["large_lesionwise_dice"],
            "best_prior_for_detection": best_det["phase"],
            "best_prior_large_lesionwise": best_det["large_lesionwise_dice"],
            "delta_large_lesionwise": best_det["large_lesionwise_dice"] - base["large_lesionwise_dice"],
            "best_detection_crosses_90pct_of_best_detector": det_crosses_best_gate,
            "clean_usable_window": clean_window,
            "loose_usable_window": loose_window,
            "fp_improvement_label": fp_label,
            "detection_improvement_label": det_label,
        })
    return pd.DataFrame(rows)


def narrative_rows(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Best per axis across all phases, plus baseline bests for context.
    fp_best = long.dropna(subset=["absent_flood_rate"]).sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
    det_best = long.dropna(subset=["large_lesionwise_dice"]).sort_values("large_lesionwise_dice", ascending=False).iloc[0]
    methods = [
        ("best_fp_restraint_any_phase", fp_best["method"]),
        ("best_detection_any_phase", det_best["method"]),
        ("baseline_best_fp_restraint", "otsu_T1c"),
        ("baseline_best_detection", "variational_spline"),
    ]
    seen = set()
    for label, method in methods:
        if (label, method) in seen:
            continue
        seen.add((label, method))
        sub = long[long["method"].eq(method)].set_index("phase")
        row = {"trace_label": label, "method": method}
        for phase in PHASES:
            if phase in sub.index:
                row[f"{phase}_flood"] = sub.loc[phase, "absent_flood_rate"]
                row[f"{phase}_median_fp"] = sub.loc[phase, "absent_median_fp_vox"]
                row[f"{phase}_large_lesionwise"] = sub.loc[phase, "large_lesionwise_dice"]
            else:
                row[f"{phase}_flood"] = np.nan
                row[f"{phase}_median_fp"] = np.nan
                row[f"{phase}_large_lesionwise"] = np.nan
        flood_vals = [row[f"{p}_flood"] for p in PHASES if np.isfinite(row[f"{p}_flood"])]
        det_vals = [row[f"{p}_large_lesionwise"] for p in PHASES if np.isfinite(row[f"{p}_large_lesionwise"])]
        row["flood_monotonic_nonincreasing"] = all(b <= a for a, b in zip(flood_vals, flood_vals[1:]))
        row["detection_monotonic_nondecreasing"] = all(b >= a for a, b in zip(det_vals, det_vals[1:]))
        rows.append(row)
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_empty_\n"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(c) for c in display.columns]
    rows = display.astype(str).values.tolist()
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def write_report(wide: pd.DataFrame, bw: pd.DataFrame, imp: pd.DataFrame, narrative: pd.DataFrame) -> None:
    path = OUT / "growth_metric_findings.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Growth Metric: Baseline to P1/P2b/P3\n\n")
        f.write("All values use the locked Stage-4 metric axes. Classical/global Dice is context only and is not used as a target here.\n\n")
        f.write("## Growth Table\n\n")
        keep_cols = ["method"]
        for phase in PHASES:
            keep_cols.extend([
                f"{phase}_absent_flood_rate",
                f"{phase}_absent_median_fp_vox",
                f"{phase}_large_lesionwise_dice",
                f"{phase}_operating_point",
            ])
        f.write(df_to_markdown(wide[keep_cols]))
        f.write("\n\n## Per-Axis Best/Worst\n\n")
        f.write(df_to_markdown(bw))
        f.write("\n\n## Did We Improve The Worst Ones?\n\n")
        f.write(df_to_markdown(imp))
        f.write("\n\nVerdict: the priors slightly improve some flood-heavy methods on the FP axis, ")
        f.write("but almost never enough to create a clean usable method. P2b is the only phase with a clean usable window, ")
        f.write("and only for `otsu_T1c`; `variational_spline` has a loose window but misses the clean median-FP bar. ")
        f.write("P3 does not beat P2b. Detection does not materially improve; the best detector remains baseline/P3 `variational_spline` at the same large lesion-wise value.\n\n")
        f.write("## Narrative Traces\n\n")
        f.write(df_to_markdown(narrative))
        f.write("\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = baseline_rows()
    long = pd.concat([
        base,
        choose_p1_rows(base),
        choose_p2b_rows(),
        p3_rows(),
    ], ignore_index=True, sort=False)
    long.to_csv(OUT / "growth_metric_long.csv", index=False)
    wide = wide_growth_table(long)
    wide.to_csv(OUT / "growth_metric_table.csv", index=False)
    bw = best_worst(long)
    bw.to_csv(OUT / "growth_metric_best_worst_by_axis.csv", index=False)
    imp = improvement_table(long)
    imp.to_csv(OUT / "growth_metric_improvement_from_baseline.csv", index=False)
    narrative = narrative_rows(long)
    narrative.to_csv(OUT / "growth_metric_narrative_traces.csv", index=False)
    write_report(wide, bw, imp, narrative)

    print("\nGROWTH TABLE")
    print(wide[[
        "method",
        "baseline_absent_flood_rate", "p1_spatial_absent_flood_rate",
        "p2b_soft_shape_absent_flood_rate", "p3_cubical_persistence_absent_flood_rate",
        "baseline_large_lesionwise_dice", "p1_spatial_large_lesionwise_dice",
        "p2b_soft_shape_large_lesionwise_dice", "p3_cubical_persistence_large_lesionwise_dice",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nIMPROVEMENT VERDICT")
    print(imp[[
        "method", "best_prior_for_fp", "delta_flood", "delta_median_fp",
        "best_prior_for_detection", "delta_large_lesionwise", "fp_improvement_label",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nReport: {OUT / 'growth_metric_findings.md'}")


if __name__ == "__main__":
    main()
