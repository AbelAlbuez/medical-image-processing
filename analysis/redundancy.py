"""
Read-only redundancy audit for the BraTS ET segmentation outputs.

This script intentionally reads existing outputs only. It does not import or
modify pipeline code, and it does not re-run segmentation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
CSV_PATH = ROOT / "output" / "tablas" / "metricas_ET.csv"
SEG_ROOT = ROOT / "output" / "segmentacion"
IMG_ROOT = ROOT / "images"
SEED = 0


def dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def read_mask(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def read_seg(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def heatmap(df: pd.DataFrame, path: Path, title: str, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(df.to_numpy(dtype=float), cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(df.shape[1]), labels=df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(df.shape[0]), labels=df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, index: bool = True) -> str:
    """Small Markdown table formatter to avoid depending on tabulate."""
    work = df.copy()
    if index:
        work = work.reset_index()
    cols = [str(c) for c in work.columns]
    rows = []
    for row in work.itertuples(index=False):
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        rows.append(vals)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for vals in rows:
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def inventory(df: pd.DataFrame) -> Tuple[List[str], List[str], pd.DataFrame]:
    cases = sorted(df["case_id"].unique())
    methods = sorted(df["metodo"].unique())
    rows = []
    for case in cases:
        gt = IMG_ROOT / case / f"{case}-seg.nii.gz"
        for method in methods:
            pred = SEG_ROOT / case / f"{case}-et_{method}.nii.gz"
            rows.append(
                {
                    "case_id": case,
                    "method": method,
                    "pred_path": str(pred.relative_to(ROOT)),
                    "pred_exists": pred.exists(),
                    "gt_path": str(gt.relative_to(ROOT)),
                    "gt_exists": gt.exists(),
                }
            )
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory_masks.csv", index=False)
    missing = inv[(~inv["pred_exists"]) | (~inv["gt_exists"])]
    if not missing.empty:
        missing.to_csv(OUT / "missing_required_outputs.csv", index=False)
        raise SystemExit(
            "Missing predicted masks or GT files. See analysis/missing_required_outputs.csv. "
            "Re-run: python run_all.py --skip-clean to regenerate segmentation masks and metrics."
        )
    return cases, methods, inv


def prediction_label_inventory(cases: Iterable[str], methods: Iterable[str]) -> pd.DataFrame:
    rows = []
    for case in cases:
        for method in methods:
            path = SEG_ROOT / case / f"{case}-et_{method}.nii.gz"
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
            vals, counts = np.unique(arr, return_counts=True)
            rows.append(
                {
                    "case_id": case,
                    "method": method,
                    "dtype": str(arr.dtype),
                    "unique_values": " ".join(str(int(v)) for v in vals),
                    "nonzero_voxels": int((arr > 0).sum()),
                    "max_value": int(arr.max()) if arr.size else 0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "prediction_label_inventory.csv", index=False)
    return out


def gt_and_metric_check(df: pd.DataFrame, cases: List[str], methods: List[str], preds) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gt_rows = []
    metric_rows = []
    lookup = df.set_index(["case_id", "metodo"])
    for case in cases:
        gt_path = IMG_ROOT / case / f"{case}-seg.nii.gz"
        seg = read_seg(gt_path)
        vals = np.unique(seg)
        gt_et = seg == 3
        gt_rows.append(
            {
                "case_id": case,
                "gt_path": str(gt_path.relative_to(ROOT)),
                "unique_labels": " ".join(str(int(v)) for v in vals),
                "et_voxels_label_3": int(gt_et.sum()),
            }
        )
        for method in methods:
            pred = preds[(case, method)]
            d = dice_bool(pred, gt_et)
            inter = np.logical_and(pred, gt_et).sum()
            union = np.logical_or(pred, gt_et).sum()
            j = 1.0 if union == 0 else float(inter / union)
            csv_row = lookup.loc[(case, method)]
            metric_rows.append(
                {
                    "case_id": case,
                    "method": method,
                    "dice_recomputed": d,
                    "dice_csv": float(csv_row["dice_ET"]),
                    "dice_abs_diff": abs(d - float(csv_row["dice_ET"])),
                    "jaccard_recomputed": j,
                    "jaccard_csv": float(csv_row["jaccard_ET"]),
                    "jaccard_abs_diff": abs(j - float(csv_row["jaccard_ET"])),
                    "vol_gt_recomputed": int(gt_et.sum()),
                    "vol_gt_csv": int(csv_row["vol_GT"]),
                    "vol_pred_recomputed": int(pred.sum()),
                    "vol_pred_csv": int(csv_row["vol_pred"]),
                }
            )
    gt_df = pd.DataFrame(gt_rows)
    metrics_df = pd.DataFrame(metric_rows)
    gt_df.to_csv(OUT / "gt_label_inventory.csv", index=False)
    metrics_df.to_csv(OUT / "metrics_recomputed_check.csv", index=False)
    return gt_df, metrics_df


def performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("metodo")
    out = g.agg(
        mean_dice=("dice_ET", "mean"),
        median_dice=("dice_ET", "median"),
        std_dice=("dice_ET", "std"),
        min_dice=("dice_ET", "min"),
        max_dice=("dice_ET", "max"),
        mean_jaccard=("jaccard_ET", "mean"),
        mean_runtime_s=("tiempo_s", "mean"),
        cases=("case_id", "count"),
    )
    out["cases_gt_0_75"] = g["dice_ET"].apply(lambda s: int((s > 0.75).sum()))
    out["cases_lt_0_25"] = g["dice_ET"].apply(lambda s: int((s < 0.25).sum()))
    out = out.sort_values("mean_dice", ascending=False).round(4)
    out.to_csv(OUT / "perf_summary.csv")
    return out


def load_predictions(cases: List[str], methods: List[str]) -> Dict[Tuple[str, str], np.ndarray]:
    preds = {}
    for case in cases:
        for method in methods:
            preds[(case, method)] = read_mask(SEG_ROOT / case / f"{case}-et_{method}.nii.gz")
    return preds


def prediction_agreement(cases: List[str], methods: List[str], preds) -> pd.DataFrame:
    mat = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            vals = [dice_bool(preds[(case, m1)], preds[(case, m2)]) for case in cases]
            mat.loc[m1, m2] = float(np.mean(vals))
    mat.to_csv(OUT / "prediction_agreement.csv")
    heatmap(mat, OUT / "prediction_agreement_heatmap.png", "Mean Dice(pred_i, pred_j)", 0, 1)
    return mat


def dendrogram_from_agreement(agreement: pd.DataFrame) -> List[dict]:
    dist = 1.0 - agreement.to_numpy(dtype=float)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(9, 5))
    dendrogram(z, labels=list(agreement.index), ax=ax, leaf_rotation=45)
    ax.set_title("Hierarchical clustering on 1 - prediction agreement")
    ax.set_ylabel("distance")
    fig.tight_layout()
    fig.savefig(OUT / "prediction_agreement_dendrogram.png", dpi=160)
    plt.close(fig)
    merges = []
    labels = list(agreement.index)
    cluster_names = {i: labels[i] for i in range(len(labels))}
    for step, row in enumerate(z, start=1):
        a, b, distance, count = int(row[0]), int(row[1]), float(row[2]), int(row[3])
        left = cluster_names[a]
        right = cluster_names[b]
        name = f"({left} + {right})"
        cluster_names[len(labels) + step - 1] = name
        merges.append({"step": step, "left": left, "right": right, "distance": distance, "n": count})
    pd.DataFrame(merges).to_csv(OUT / "prediction_agreement_cluster_tree.csv", index=False)
    return merges


def performance_correlation(df: pd.DataFrame, cases: List[str], methods: List[str]) -> pd.DataFrame:
    pivot = df.pivot(index="case_id", columns="metodo", values="dice_ET").loc[cases, methods]
    mat = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            rho, _ = spearmanr(pivot[m1], pivot[m2])
            mat.loc[m1, m2] = float(rho) if not math.isnan(rho) else np.nan
    mat.to_csv(OUT / "performance_spearman.csv")
    heatmap(mat, OUT / "performance_spearman_heatmap.png", "Spearman correlation of per-case Dice", -1, 1)
    return mat


def double_fault(df: pd.DataFrame, cases: List[str], methods: List[str]) -> pd.DataFrame:
    pivot = df.pivot(index="case_id", columns="metodo", values="dice_ET").loc[cases, methods]
    mat = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            mat.loc[m1, m2] = float(((pivot[m1] < 0.5) & (pivot[m2] < 0.5)).mean())
    mat.to_csv(OUT / "double_fault.csv")
    heatmap(mat, OUT / "double_fault_heatmap.png", "Fraction of cases where both Dice < 0.5", 0, 1)
    return mat


def oracle_analysis(df: pd.DataFrame, cases: List[str], methods: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivot = df.pivot(index="case_id", columns="metodo", values="dice_ET").loc[cases, methods]
    best_single_method = pivot.mean(axis=0).idxmax()
    best_single_mean = float(pivot.mean(axis=0).max())
    oracle = pivot.max(axis=1)
    winners = []
    sole_best_rows = []
    sole_over_rows = []
    for case, row in pivot.iterrows():
        maxv = row.max()
        bests = list(row[row == maxv].index)
        winners.append({"case_id": case, "oracle_dice": maxv, "winner_methods": ";".join(bests), "n_winners": len(bests)})
        if len(bests) == 1:
            sole_best_rows.append({"case_id": case, "method": bests[0], "dice": maxv})
        above = list(row[row > 0.75].index)
        for method in above:
            if len(above) == 1:
                sole_over_rows.append({"case_id": case, "method": method, "dice": row[method]})
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(OUT / "oracle_by_case.csv", index=False)
    pd.DataFrame(sole_best_rows).to_csv(OUT / "sole_best_cases.csv", index=False)
    pd.DataFrame(sole_over_rows).to_csv(OUT / "sole_above_075_cases.csv", index=False)

    full_oracle = float(oracle.mean())
    loo_rows = []
    for method in methods:
        without = float(pivot.drop(columns=[method]).max(axis=1).mean())
        loo_rows.append(
            {
                "method": method,
                "oracle_without_method": without,
                "delta_full_minus_without": full_oracle - without,
                "sole_best_cases": int((pd.DataFrame(sole_best_rows)["method"] == method).sum())
                if sole_best_rows
                else 0,
                "sole_above_075_cases": int((pd.DataFrame(sole_over_rows)["method"] == method).sum())
                if sole_over_rows
                else 0,
            }
        )
    loo = pd.DataFrame(loo_rows).sort_values("delta_full_minus_without", ascending=False)
    loo.to_csv(OUT / "oracle_leave_one_out.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "cases": len(cases),
                "methods": len(methods),
                "best_single_method": best_single_method,
                "best_single_mean_dice": best_single_mean,
                "oracle_mean_dice": full_oracle,
                "headroom_oracle_minus_best_single": full_oracle - best_single_mean,
            }
        ]
    )
    summary.to_csv(OUT / "oracle_summary.csv", index=False)
    return summary, loo


def identical_mask_report(cases: List[str], methods: List[str], preds) -> pd.DataFrame:
    rows = []
    for case in cases:
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1 :]:
                same = bool(np.array_equal(preds[(case, m1)], preds[(case, m2)]))
                if same:
                    rows.append({"case_id": case, "method_a": m1, "method_b": m2})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "identical_masks_by_case.csv", index=False)
    return out


def write_report(
    df: pd.DataFrame,
    methods: List[str],
    cases: List[str],
    inv: pd.DataFrame,
    labels: pd.DataFrame,
    perf: pd.DataFrame,
    agreement: pd.DataFrame,
    corr: pd.DataFrame,
    faults: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    loo: pd.DataFrame,
    identical: pd.DataFrame,
    merges: List[dict],
    gt_labels: pd.DataFrame,
    metric_check: pd.DataFrame,
) -> None:
    top_agree = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            top_agree.append((m1, m2, float(agreement.loc[m1, m2])))
    top_agree = sorted(top_agree, key=lambda x: x[2], reverse=True)[:10]
    top_corr = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            top_corr.append((m1, m2, float(corr.loc[m1, m2])))
    top_corr = sorted(top_corr, key=lambda x: x[2], reverse=True)[:10]
    top_fault = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            top_fault.append((m1, m2, float(faults.loc[m1, m2])))
    top_fault = sorted(top_fault, key=lambda x: x[2], reverse=True)[:10]

    method_inventory = pd.DataFrame(
        [
            {
                "method": "otsu_T1c",
                "source": "src/brats_pipeline/seg_et_pipeline.py:81",
                "signals": "cleaned T1c",
                "family": "intensity-threshold",
                "key_params": "threshold_multiotsu classes=3; choose brightest class; morphology erosion=1 dilation=1 keep_largest=True",
                "shared_dependencies": "cleaned T1c; _morfologia",
            },
            {
                "method": "gmm_T1c",
                "source": "src/brats_pipeline/seg_et_pipeline.py:96",
                "signals": "cleaned T1c",
                "family": "mixture-model",
                "key_params": "GaussianMixture n_components=3 random_state=42; choose highest-mean cluster; morphology erosion=1 dilation=1 keep_largest=True",
                "shared_dependencies": "also used as roi_et_auto primary seed",
            },
            {
                "method": "sustraccion",
                "source": "src/brats_pipeline/seg_et_pipeline.py:117",
                "signals": "raw T1c, raw T1n -> mapa_dif",
                "family": "enhancement-threshold",
                "key_params": "joint normalization p=99.5; gaussian sigma=0.5; threshold positive mapa_dif at auto_pct=90; morphology keep_largest=False",
                "shared_dependencies": "_mapa_diferencia",
            },
            {
                "method": "gmm_2d",
                "source": "src/brats_pipeline/seg_et_pipeline.py:132",
                "signals": "cleaned T1c + mapa_dif",
                "family": "mixture-model",
                "key_params": "GaussianMixture n_components=4 random_state=42 max_iter=200; reject dif_mean>0.40 or t1c_mean<0.8*median; score=0.4*T1c+0.6*dif",
                "shared_dependencies": "_mapa_diferencia; cleaned T1c",
            },
            {
                "method": "rango_doble",
                "source": "src/brats_pipeline/seg_et_pipeline.py:185",
                "signals": "raw T1c, raw T1n -> mapa_dif",
                "family": "enhancement-threshold",
                "key_params": "lower threshold=70th percentile of positive mapa_dif; upper threshold=0.55; remove components <30 vox",
                "shared_dependencies": "_mapa_diferencia",
            },
            {
                "method": "fast_marching",
                "source": "src/brats_pipeline/seg_et_pipeline.py:233",
                "signals": "raw T1c, raw T1n -> mapa_dif; automatic or manual seed",
                "family": "seed/front",
                "key_params": "auto seed score=T1c*mapa_dif with 0.05<mapa<0.55 and uniform_filter size=5; tiempo_umbral=35.0; sigma=0.8",
                "shared_dependencies": "_mapa_diferencia; semilla_automatica",
            },
            {
                "method": "semilla",
                "source": "src/brats_pipeline/seg_et_pipeline.py:318",
                "signals": "raw T1c, raw T1n -> mapa_dif; manual seed",
                "family": "seed/front",
                "key_params": "sphere radius=25; local percentile=65; morphology erosion=1 dilation=2 keep_largest=True",
                "shared_dependencies": "requires semilla_zyx; did not run in current CSV",
            },
            {
                "method": "level_set",
                "source": "src/brats_pipeline/seg_spline_levelset.py:212",
                "signals": "cleaned T1c + raw T1c/T1n through shared roi_et_auto/mapa_dif",
                "family": "deformable-contour",
                "key_params": "GAC prop=0.8 curv=3.0 adv=1.5 iters=120; sigmoid gradient alpha=-0.05 beta=0.1; bbox margin=12",
                "shared_dependencies": "shared roi_et_auto and _post with other deformables",
            },
            {
                "method": "variational_spline",
                "source": "src/brats_pipeline/seg_spline_levelset.py:268",
                "signals": "shared roi_et_auto/mapa_dif",
                "family": "deformable-contour",
                "key_params": "morphological_chan_vese num_iter=35 smoothing=3 lambda1=lambda2=1.0; bbox margin=12",
                "shared_dependencies": "shared roi_et_auto and _post with other deformables",
            },
            {
                "method": "bspline",
                "source": "src/brats_pipeline/seg_spline_levelset.py:333",
                "signals": "shared roi_et_auto/mapa_dif",
                "family": "deformable-contour",
                "key_params": "Chan-Vese num_iter=35 smoothing=2; per-slice cubic periodic B-spline smooth=3.0; pct_realce=78",
                "shared_dependencies": "shared roi_et_auto and _post with other deformables; starts from Chan-Vese-like mask",
            },
            {
                "method": "spline",
                "source": "src/brats_pipeline/seg_spline_levelset.py:378",
                "signals": "shared roi_et_auto/mapa_dif",
                "family": "deformable-contour",
                "key_params": "active_contour alpha=0.05 beta=2.0 w_line=2.0 w_edge=1.0 gamma=0.02 max_iter=25; bbox margin=8",
                "shared_dependencies": "shared roi_et_auto and _post with other deformables",
            },
        ]
    )
    method_inventory.to_csv(OUT / "method_inventory.csv", index=False)

    report = []
    report.append("# BraTS ET Redundancy Audit\n")
    report.append("## Phase 0 - Reconnaissance\n")
    report.append(f"- Metrics CSV: `{CSV_PATH.relative_to(ROOT)}` with {len(df)} rows, {len(cases)} cases, {len(methods)} methods.\n")
    report.append(f"- Predicted masks expected and present: {int(inv['pred_exists'].sum())}/{len(inv)}.\n")
    report.append("- Naming convention: `output/segmentacion/<case>/<case>-et_<method>.nii.gz`.\n")
    report.append("- Prediction label convention: binary masks; see `prediction_label_inventory.csv` (`unique_values` is `0 1` for persisted masks).\n")
    report.append("- GT location: `images/<case>/<case>-seg.nii.gz`; ET is extracted as label `3` in code.\n")
    report.append(f"- GT labels observed on disk: {sorted(set(' '.join(gt_labels['unique_labels']).split()))}; see `gt_label_inventory.csv`.\n")
    report.append(f"- Recomputed-vs-CSV max Dice abs diff: {metric_check['dice_abs_diff'].max():.6f}; max Jaccard abs diff: {metric_check['jaccard_abs_diff'].max():.6f}; see `metrics_recomputed_check.csv`.\n")
    report.append(f"- Methods in CSV: {', '.join(methods)}.\n")
    not_run = sorted(set(method_inventory["method"]) - set(methods))
    report.append(f"- Method in code but not present in this run: {', '.join(not_run) if not_run else 'none'}.\n")
    report.append("\n## Phase 1 - Method Inventory\n")
    report.append(markdown_table(method_inventory, index=False))
    report.append("\n\nA-priori redundancy hypotheses before looking at mask agreement:\n")
    report.append("- `level_set`, `variational_spline`, `bspline`, and `spline` should be correlated because they share `roi_et_auto`, `mapa_dif`, and `_post` safeguards.\n")
    report.append("- `gmm_T1c` should be close to deformables whenever the shared ROI falls back to or accepts the GMM seed.\n")
    report.append("- `sustraccion` and `rango_doble` should share enhancement-map blind spots, but not necessarily identical masks because one is a one-sided high percentile and the other is a bounded interval.\n")
    report.append("- `fast_marching` may be complementary if the auto seed lands well, but vulnerable to seed/front leakage or undergrowth.\n")
    report.append("\n## Phase 2 - Quantitative Redundancy\n")
    report.append("\n### Performance Summary\n")
    report.append(markdown_table(perf))
    report.append("\n\n### Top Prediction-Agreement Pairs\n")
    report.extend([f"- {a} vs {b}: {v:.4f}\n" for a, b, v in top_agree])
    report.append("\n### Top Performance-Correlation Pairs\n")
    report.extend([f"- {a} vs {b}: {v:.4f}\n" for a, b, v in top_corr])
    report.append("\n### Top Double-Fault Pairs\n")
    report.extend([f"- {a} vs {b}: {v:.4f}\n" for a, b, v in top_fault])
    report.append("\n### Oracle\n")
    report.append(markdown_table(oracle_summary, index=False))
    report.append("\n\n### Leave-One-Out Unique Value\n")
    report.append(markdown_table(loo, index=False))
    report.append("\n\n### Cluster Tree\n")
    for m in merges:
        report.append(f"- step {m['step']}: {m['left']} + {m['right']} at distance {m['distance']:.4f} (n={m['n']})\n")
    report.append("\n### Surprises / Integrity Flags\n")
    if identical.empty:
        report.append("- No exactly identical mask pairs were found on a per-case basis.\n")
    else:
        report.append(f"- Found {len(identical)} exactly identical per-case mask pairs; see `identical_masks_by_case.csv`.\n")
    high_agree = [(a, b, v) for a, b, v in top_agree if v > 0.9]
    if high_agree:
        report.append("- Very high mean prediction agreement (>0.90): " + "; ".join(f"{a}/{b}={v:.3f}" for a, b, v in high_agree) + ".\n")
    else:
        report.append("- No method pair exceeded 0.90 mean prediction agreement.\n")
    if perf["mean_runtime_s"].nunique() == 1:
        report.append("- Runtime is identical for every method in the CSV. Code assigns one case-level elapsed time to all method rows, so `tiempo_s` is not per-method runtime (`run_all.py:168`).\n")
    report.append("\n## Artifacts\n")
    for name in sorted(p.name for p in OUT.iterdir() if p.name != "redundancy.py"):
        report.append(f"- `{name}`\n")
    (OUT / "report.md").write_text("".join(report), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    if not CSV_PATH.exists():
        raise SystemExit("Missing output/tablas/metricas_ET.csv")
    df = pd.read_csv(CSV_PATH)
    numeric_cols = ["dice_ET", "jaccard_ET", "vol_GT", "vol_pred", "tiempo_s"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    cases, methods, inv = inventory(df)
    labels = prediction_label_inventory(cases, methods)
    perf = performance_summary(df)
    preds = load_predictions(cases, methods)
    gt_labels, metric_check = gt_and_metric_check(df, cases, methods, preds)
    identical = identical_mask_report(cases, methods, preds)
    agreement = prediction_agreement(cases, methods, preds)
    merges = dendrogram_from_agreement(agreement)
    corr = performance_correlation(df, cases, methods)
    faults = double_fault(df, cases, methods)
    oracle_summary, loo = oracle_analysis(df, cases, methods)
    metadata = {
        "seed": SEED,
        "root": str(ROOT),
        "cases": cases,
        "methods": methods,
        "inputs": {
            "metrics_csv": str(CSV_PATH.relative_to(ROOT)),
            "segmentation_root": str(SEG_ROOT.relative_to(ROOT)),
            "gt_root": str(IMG_ROOT.relative_to(ROOT)),
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        df, methods, cases, inv, labels, perf, agreement, corr, faults,
        oracle_summary, loo, identical, merges, gt_labels, metric_check
    )
    print(f"Wrote redundancy audit artifacts to {OUT}")


if __name__ == "__main__":
    main()
