from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
BASELINE = ANALYSIS / "baseline"
DEFORMABLES = ["bspline", "level_set", "spline", "variational_spline"]


def read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def compare_means() -> pd.DataFrame:
    old = pd.read_csv(BASELINE / "old_baseline_means.csv").rename(
        columns={"mean_dice": "old_mean_dice"}
    )
    new = pd.read_csv(BASELINE / "baseline_means.csv").rename(
        columns={"mean_dice": "new_mean_dice"}
    )
    out = old.merge(new, on="metodo", how="outer")
    out["delta_new_minus_old"] = out["new_mean_dice"] - out["old_mean_dice"]
    out = out.sort_values("new_mean_dice", ascending=False)
    out.to_csv(BASELINE / "old_vs_new_mean_dice.csv", index=False)
    return out


def compare_prediction_agreement() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = read_matrix(BASELINE / "old_prediction_agreement.csv")
    new = read_matrix(ANALYSIS / "prediction_agreement.csv")
    methods = sorted(set(old.index) & set(new.index))
    rows = []
    for a, b in combinations(methods, 2):
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "old_agreement": float(old.loc[a, b]),
                "new_agreement": float(new.loc[a, b]),
                "delta_new_minus_old": float(new.loc[a, b] - old.loc[a, b]),
            }
        )
    pairwise = pd.DataFrame(rows).sort_values("delta_new_minus_old")
    pairwise.to_csv(ANALYSIS / "prediction_agreement_old_vs_new.csv", index=False)

    def offdiag_mean(mat: pd.DataFrame, subset: list[str]) -> float:
        vals = [float(mat.loc[a, b]) for a, b in combinations(subset, 2)]
        return float(np.mean(vals))

    deformables = [m for m in DEFORMABLES if m in methods]
    summary = pd.DataFrame(
        [
            {
                "group": "deformables_offdiag",
                "old_mean_agreement": offdiag_mean(old, deformables),
                "new_mean_agreement": offdiag_mean(new, deformables),
            },
            {
                "group": "all_methods_offdiag",
                "old_mean_agreement": offdiag_mean(old, methods),
                "new_mean_agreement": offdiag_mean(new, methods),
            },
        ]
    )
    summary["delta_new_minus_old"] = (
        summary["new_mean_agreement"] - summary["old_mean_agreement"]
    )
    summary.to_csv(ANALYSIS / "prediction_agreement_change_summary.csv", index=False)
    return pairwise, summary


def guard_split() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "output" / "tablas" / "metricas_ET.csv")
    rows = []
    for method, sub in df.groupby("metodo"):
        evolved = sub[sub["guard_branch"] == "evolved"]
        fallback = sub[sub["guard_branch"].isin(["ROI-fallback", "collapse-detected"])]
        rows.append(
            {
                "method": method,
                "n_evolved": int(len(evolved)),
                "mean_dice_evolved": (
                    float(evolved["dice_ET"].mean()) if len(evolved) else np.nan
                ),
                "n_fallback": int(len(fallback)),
                "mean_dice_fallback": (
                    float(fallback["dice_ET"].mean()) if len(fallback) else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values("method")
    out.to_csv(ANALYSIS / "guard_split_performance.csv", index=False)
    return out


def keep_cut() -> pd.DataFrame:
    perf = pd.read_csv(ANALYSIS / "perf_summary.csv")
    loo = pd.read_csv(ANALYSIS / "oracle_leave_one_out.csv")
    out = loo.merge(perf[["metodo", "mean_dice"]], left_on="method", right_on="metodo")
    out = out.drop(columns=["metodo"])
    out["recommendation"] = np.where(
        (out["delta_full_minus_without"] > 0.001) | (out["sole_best_cases"] > 0),
        "KEEP",
        "CUT",
    )
    out["rationale"] = np.where(
        out["recommendation"] == "KEEP",
        "positive leave-one-out oracle value or sole-best case",
        "zero leave-one-out delta and no sole-best cases",
    )
    out = out.sort_values(
        ["recommendation", "delta_full_minus_without", "mean_dice"],
        ascending=[False, False, False],
    )
    out.to_csv(ANALYSIS / "keep_cut_updated.csv", index=False)
    return out


def main() -> None:
    means = compare_means()
    _, agreement_summary = compare_prediction_agreement()
    split = guard_split()
    table = keep_cut()
    print("old_vs_new_mean_dice")
    print(means.round(4).to_string(index=False))
    print("\nprediction_agreement_change_summary")
    print(agreement_summary.round(4).to_string(index=False))
    print("\nguard_split_performance")
    print(split.round(4).to_string(index=False))
    print("\nkeep_cut_updated")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
