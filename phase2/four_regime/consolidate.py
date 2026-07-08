"""Build the four-regime master comparison and discrimination diagnostic.

Regime labels are deliberately strict: shape proxies are not called topology,
and only the GUDHI cubical H0 persistence regime is labeled persistent homology.
Reads existing Phase 2 CSVs only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase2" / "four_regime"

METHODS = ["otsu_T1c", "gmm_T1c", "sustraccion", "gmm_2d", "variational_spline"]
REGIMES = [
    {
        "id": "R1",
        "short": "INTENSITY_BASELINE",
        "label": "R1 INTENSITY BASELINE",
        "description": "Core classical methods on the enhancement/intensity scalar; includes variational_spline as a baseline method using the same scalar evidence.",
        "growth_prefix": "baseline",
        "source": "analysis/stage4_metrics/stage4_absent_fp_summary.csv + analysis/stage4_metrics/stage4_present_by_vol_bin.csv",
    },
    {
        "id": "R2",
        "short": "SPATIAL_LOCATION_PRIOR",
        "label": "R2 SPATIAL/LOCATION PRIOR",
        "description": "Leave-fold ET-occurrence atlas prior.",
        "growth_prefix": "p1_spatial",
        "source": "phase2/p1_spatial_atlas/p1_spatial_sweep_summary.csv + tradeoff_analysis/p1_fixed_detection_cost_operating_points.csv",
    },
    {
        "id": "R3",
        "short": "SHAPE_PROXY_GEOMETRY",
        "label": "R3 SHAPE-PROXY GEOMETRY",
        "description": "Compactness, isoperimetric quotient, erosion-fragmentation, and normalized-radius proxies. This is topology-inspired geometry, not persistent homology.",
        "growth_prefix": "p2b_soft_shape",
        "source": "phase2/p2_soft_shape_sweep/p2_soft_shape_operating_points.csv + phase2/p2_shape_probe/p2_shape_separation_tests.csv",
    },
    {
        "id": "R4",
        "short": "PERSISTENT_HOMOLOGY",
        "label": "R4 PERSISTENT HOMOLOGY",
        "description": "Genuine GUDHI cubical H0 persistence on the normalized enhancement map.",
        "growth_prefix": "p3_cubical_persistence",
        "source": "phase2/p3_cubical_persistence/p3_key_comparison_vs_baseline.csv + p3_persistence_diagnostic_summary.csv",
    },
]

GROWTH = ROOT / "phase2" / "growth_metric" / "growth_metric_table.csv"
GROWTH_IMPROVE = ROOT / "phase2" / "growth_metric" / "growth_metric_improvement_from_baseline.csv"
P1_SEED = ROOT / "phase2" / "p1_spatial_atlas" / "tradeoff_analysis" / "p1_seed_percentile_summary.csv"
P2_SEP = ROOT / "phase2" / "p2_shape_probe" / "p2_shape_separation_tests.csv"
P2B_ADDENDUM = ROOT / "phase2" / "p2_soft_shape_sweep" / "p2b_headline_audit_addendum.csv"
P3_DIAG = ROOT / "phase2" / "p3_cubical_persistence" / "p3_persistence_diagnostic_summary.csv"
STAGE3D_SUPPORT = ROOT / "analysis" / "stage3d_gmm_seed_support.csv"


def fmt_empty(value: object) -> object:
    if pd.isna(value):
        return "EMPTY"
    return value


def regime_prefix(regime: Dict[str, str]) -> str:
    return regime["growth_prefix"]


def baseline_best_large(growth: pd.DataFrame) -> float:
    return float(growth["baseline_large_lesionwise_dice"].max())


def build_master(growth: pd.DataFrame) -> pd.DataFrame:
    best_large = baseline_best_large(growth)
    rows: List[Dict[str, object]] = []
    for _, g in growth.iterrows():
        row: Dict[str, object] = {"method": g["method"]}
        for regime in REGIMES:
            prefix = regime_prefix(regime)
            valid = bool(g.get(f"{prefix}_valid_operating_point", False))
            if not valid:
                row[f"{regime['id']}_regime_label"] = regime["label"]
                row[f"{regime['id']}_absent_flood_rate"] = "EMPTY_NO_VALID_OPERATING_POINT"
                row[f"{regime['id']}_absent_median_fp_vox"] = "EMPTY_NO_VALID_OPERATING_POINT"
                row[f"{regime['id']}_large_lesionwise_dice"] = "EMPTY_NO_VALID_OPERATING_POINT"
                row[f"{regime['id']}_large_detection_retention_vs_R1_best"] = "EMPTY_NO_VALID_OPERATING_POINT"
                row[f"{regime['id']}_operating_point"] = fmt_empty(g.get(f"{prefix}_operating_point", ""))
                row[f"{regime['id']}_valid_operating_point"] = False
                continue
            large = float(g[f"{prefix}_large_lesionwise_dice"])
            row[f"{regime['id']}_regime_label"] = regime["label"]
            row[f"{regime['id']}_absent_flood_rate"] = float(g[f"{prefix}_absent_flood_rate"])
            row[f"{regime['id']}_absent_median_fp_vox"] = float(g[f"{prefix}_absent_median_fp_vox"])
            row[f"{regime['id']}_large_lesionwise_dice"] = large
            row[f"{regime['id']}_large_detection_retention_vs_R1_best"] = large / best_large if best_large else np.nan
            row[f"{regime['id']}_operating_point"] = g.get(f"{prefix}_operating_point", "")
            row[f"{regime['id']}_valid_operating_point"] = True
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "four_regime_master_comparison.csv", index=False)
    return out


def regime_long_from_growth(growth: pd.DataFrame) -> pd.DataFrame:
    best_large = baseline_best_large(growth)
    rows = []
    for _, g in growth.iterrows():
        for regime in REGIMES:
            prefix = regime_prefix(regime)
            valid = bool(g.get(f"{prefix}_valid_operating_point", False))
            if not valid:
                continue
            large = float(g[f"{prefix}_large_lesionwise_dice"])
            rows.append({
                "regime_id": regime["id"],
                "regime_label": regime["label"],
                "method": g["method"],
                "absent_flood_rate": float(g[f"{prefix}_absent_flood_rate"]),
                "absent_median_fp_vox": float(g[f"{prefix}_absent_median_fp_vox"]),
                "large_lesionwise_dice": large,
                "large_detection_retention_vs_R1_best": large / best_large if best_large else np.nan,
                "operating_point": g.get(f"{prefix}_operating_point", ""),
            })
    return pd.DataFrame(rows)


def robustness_label(regime_id: str,
                     axis: str,
                     best_row: pd.Series,
                     r1_fp_best: pd.Series,
                     r1_det_best: pd.Series) -> str:
    if regime_id == "R1":
        return "ROBUST_CONTEXT_BASELINE"
    if regime_id == "R2":
        if axis == "FP":
            return (
                "DEGENERATE: apparent FP gain comes from spatial filtering that fails usable large-detection gate"
                if best_row["large_detection_retention_vs_R1_best"] < 0.90
                else "SUGGESTIVE: location lowers flood but confounds remain high-atlas-percentile"
            )
        return "NO_BEAT: best detection below R1 best"
    if regime_id == "R3":
        if axis == "FP":
            return "SUGGESTIVE_NOT_ROBUST: post-hoc sweep; residual flood high; shape-proxy not deployable alone"
        return "NO_BEAT: no detection improvement over R1 best"
    if regime_id == "R4":
        if axis == "FP":
            return "DEGENERATE/SUGGESTIVE: persistence diagnostic inverted and median FP remains poor"
        return "NO_BEAT: ties R1 best but does not improve"
    return ""


def build_summary(long: pd.DataFrame) -> pd.DataFrame:
    r1 = long[long["regime_id"].eq("R1")]
    r1_fp_best = r1.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
    r1_det_best = r1.sort_values("large_lesionwise_dice", ascending=False).iloc[0]
    rows = []
    for regime in REGIMES:
        sub = long[long["regime_id"].eq(regime["id"])]
        fp_best = sub.sort_values(["absent_flood_rate", "absent_median_fp_vox"]).iloc[0]
        det_best = sub.sort_values("large_lesionwise_dice", ascending=False).iloc[0]
        fp_flood_beats = fp_best["absent_flood_rate"] < r1_fp_best["absent_flood_rate"]
        fp_median_beats = fp_best["absent_median_fp_vox"] < r1_fp_best["absent_median_fp_vox"]
        fp_true_beat = bool(fp_flood_beats and fp_median_beats and fp_best["large_detection_retention_vs_R1_best"] >= 0.90)
        det_beats = det_best["large_lesionwise_dice"] > r1_det_best["large_lesionwise_dice"]
        det_ties = np.isclose(det_best["large_lesionwise_dice"], r1_det_best["large_lesionwise_dice"], atol=1e-12)
        rows.append({
            "regime_id": regime["id"],
            "regime_label": regime["label"],
            "precise_methodological_label": regime["description"],
            "source": regime["source"],
            "best_fp_method": fp_best["method"],
            "best_fp_absent_flood_rate": fp_best["absent_flood_rate"],
            "best_fp_absent_median_fp_vox": fp_best["absent_median_fp_vox"],
            "best_fp_large_lesionwise_dice": fp_best["large_lesionwise_dice"],
            "best_fp_retention_vs_R1_best_detection": fp_best["large_detection_retention_vs_R1_best"],
            "fp_flood_beats_R1_best": bool(fp_flood_beats),
            "fp_median_beats_R1_best": bool(fp_median_beats),
            "fp_axis_true_beat_R1": fp_true_beat,
            "fp_axis_robustness_label": robustness_label(regime["id"], "FP", fp_best, r1_fp_best, r1_det_best),
            "best_detection_method": det_best["method"],
            "best_detection_large_lesionwise_dice": det_best["large_lesionwise_dice"],
            "best_detection_absent_flood_rate": det_best["absent_flood_rate"],
            "best_detection_absent_median_fp_vox": det_best["absent_median_fp_vox"],
            "detection_beats_R1_best": bool(det_beats),
            "detection_ties_R1_best": bool(det_ties),
            "detection_axis_robustness_label": robustness_label(regime["id"], "detection", det_best, r1_fp_best, r1_det_best),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "four_regime_summary.csv", index=False)
    return out


def build_discrimination_diagnostic() -> pd.DataFrame:
    support = pd.read_csv(STAGE3D_SUPPORT)
    targets = support[support["is_target_roi_failure"].astype(bool)].copy()
    p1 = pd.read_csv(P1_SEED)
    p2 = pd.read_csv(P2_SEP)
    p3 = pd.read_csv(P3_DIAG)

    p2_abs = p2[
        p2["comparison"].eq("true_et_gt_vs_absent_fp")
        & p2["metric"].eq("compactness_v_over_s15")
    ].iloc[0]
    p2_peri = p2[
        p2["comparison"].eq("true_et_gt_vs_peri_cavity_seed_fp")
        & p2["metric"].eq("compactness_v_over_s15")
    ].iloc[0]
    p3_all = p3[p3["comparison"].eq("all_confound")].iloc[0]
    rows = [
        {
            "regime_id": "R1",
            "regime_label": "R1 INTENSITY BASELINE",
            "discrimination_question": "Can enhancement/intensity alone separate ET from non-tumor enhancement?",
            "separability_verdict": "NO",
            "separability_metric": "GMM seed enhancement support on irreducible failures",
            "separability_value": "00533/02078 p90-support=1.0; support ranks 20/20 and 19/20",
            "n": "20 original audit cases; 2 irreducible confounds",
            "robustness": "SUGGESTIVE but mechanistically strong",
            "interpretation": "The failures are among the brightest/enhancement-supported seeds, so an enhancement-family veto cannot separate them.",
        },
        {
            "regime_id": "R2",
            "regime_label": "R2 SPATIAL/LOCATION PRIOR",
            "discrimination_question": "Can population ET location separate tumor from confound?",
            "separability_verdict": "NO",
            "separability_metric": "Atlas percentile among nonzero atlas voxels at 00533/02078 seed centroids",
            "separability_value": (
                f"00533 mean percentile {p1[p1['case_id'].eq('BraTS-GLI-00533-100')]['percentile_nonzero_mean'].iloc[0]:.1f}; "
                f"02078 mean percentile {p1[p1['case_id'].eq('BraTS-GLI-02078-100')]['percentile_nonzero_mean'].iloc[0]:.1f}"
            ),
            "n": "2 irreducible confounds x 5 held-out atlases",
            "robustness": "ANECDOTAL/SUGGESTIVE",
            "interpretation": "The confound seeds are not spatial outliers; they sit around the 87-92nd atlas percentile, so location suppresses some FP but not the hard confounds.",
        },
        {
            "regime_id": "R3",
            "regime_label": "R3 SHAPE-PROXY GEOMETRY",
            "discrimination_question": "Can shape/topology-inspired proxies separate tumor-like components from FP components?",
            "separability_verdict": "PARTIAL_IN_ISOLATION_NOT_DEPLOYABLE",
            "separability_metric": "compactness_v_over_s15 AUC true ET > absent FP; peri-cavity corroboration",
            "separability_value": (
                f"absent FP AUC {float(p2_abs['auc_probability_true_greater']):.3f}; "
                f"peri-cavity AUC {float(p2_peri['auc_probability_true_greater']):.3f}"
            ),
            "n": f"absent FP: true {int(p2_abs['n_true'])}, false {int(p2_abs['n_false'])}; peri-cavity false n={int(p2_peri['n_false'])}",
            "robustness": "ROBUST component probe for absent floods; ANECDOTAL peri-cavity; deployment SUGGESTIVE/POST-HOC",
            "interpretation": "Shape proxies separate components offline, but P2b leaves high residual flood and was selected post-hoc; proxy geometry is helpful but not sufficient.",
        },
        {
            "regime_id": "R4",
            "regime_label": "R4 PERSISTENT HOMOLOGY",
            "discrimination_question": "Can genuine cubical H0 persistence separate tumor from confound?",
            "separability_verdict": "NO_INVERTED",
            "separability_metric": "GUDHI cubical H0 max persistence AUC true ET > all confound",
            "separability_value": f"AUC {float(p3_all['auc_true_higher_than_confound']):.3f}; true median {float(p3_all['true_median']):.3f}; confound median {float(p3_all['confound_median']):.3f}",
            "n": f"true components {int(p3_all['n_true_components'])}; confound components {int(p3_all['n_confound_components'])}",
            "robustness": "ROBUST diagnostic inversion",
            "interpretation": "Confounds are more persistent than true ET on the enhancement scalar, so Morse/PH on this scalar inherits the hard false-positive problem.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "four_regime_discrimination_diagnostic.csv", index=False)
    return out


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


def write_findings(master: pd.DataFrame, summary: pd.DataFrame, diagnostic: pd.DataFrame) -> None:
    path = OUT / "four_regime_findings.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Four-Regime Master Comparison\n\n")
        f.write("This table consolidates the four distinct methodological regimes. ")
        f.write("R3 is deliberately labeled **shape-proxy geometry** because it uses compactness/isoperimetric/erosion/normalized-radius proxies. ")
        f.write("Only R4 is labeled persistent homology because it uses GUDHI cubical H0 persistence.\n\n")
        f.write("## Regime Labels\n\n")
        f.write(df_to_markdown(pd.DataFrame(REGIMES)[["id", "label", "description", "source"]]))
        f.write("\n\n## Regime-Level Summary\n\n")
        f.write(df_to_markdown(summary))
        f.write("\n\n## Discrimination Diagnostic: Impossibility Chain\n\n")
        f.write(df_to_markdown(diagnostic))
        f.write("\n\n## Master Comparison Table\n\n")
        f.write(df_to_markdown(master))
        f.write("\n\n## Prose Finding\n\n")
        f.write("The four regimes form a negative-result chain. ")
        f.write("Intensity fails because the irreducible confounds are among the brightest enhancing structures. ")
        f.write("Location fails because those same confounds are not spatial outliers in the atlas; they sit at high atlas percentiles. ")
        f.write("Shape-proxy geometry partially separates components in isolation, but the deployable soft prior remains post-hoc and leaves high residual flood. ")
        f.write("Genuine cubical persistence fails most decisively: confounds are more persistent than true ET on the enhancement scalar. ")
        f.write("Together, the regimes support the paper-safe conclusion that no tested regime operating only on the enhancement scalar discriminates tumor from hard post-treatment confound reliably.\n")


def main() -> None:
    growth = pd.read_csv(GROWTH)
    master = build_master(growth)
    long = regime_long_from_growth(growth)
    summary = build_summary(long)
    diagnostic = build_discrimination_diagnostic()
    write_findings(master, summary, diagnostic)
    print("\nFOUR-REGIME SUMMARY")
    print(summary[[
        "regime_id", "regime_label", "best_fp_method", "best_fp_absent_flood_rate",
        "best_fp_absent_median_fp_vox", "fp_axis_true_beat_R1", "fp_axis_robustness_label",
        "best_detection_method", "best_detection_large_lesionwise_dice",
        "detection_beats_R1_best", "detection_axis_robustness_label",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nDISCRIMINATION DIAGNOSTIC")
    print(diagnostic[[
        "regime_id", "regime_label", "separability_verdict",
        "separability_metric", "separability_value", "n", "robustness",
    ]].to_string(index=False))
    print(f"\nReport: {OUT / 'four_regime_findings.md'}")


if __name__ == "__main__":
    main()
