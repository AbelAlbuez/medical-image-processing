from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 0
BOOTSTRAPS = 2000

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "phase2" / "final_data" / "canonical"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from phase2.metrics import DEFAULT_CONFIG  # noqa: E402


def version_of(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"MISSING ({exc})"


def print_versions() -> None:
    versions = {
        "numpy": np.__version__,
        "scipy": version_of("scipy"),
        "sklearn": version_of("sklearn"),
        "gudhi": version_of("gudhi"),
    }
    print("VERSIONS " + json.dumps(versions, sort_keys=True))


def read_csv(path: Path, missing: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        missing.append(str(path.relative_to(ROOT)))
        return None
    return pd.read_csv(path)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def bootstrap_ci_case_clustered(
    df: pd.DataFrame,
    case_col: str,
    reducer,
    n: int = BOOTSTRAPS,
) -> tuple[float, float, float]:
    cases = np.array(sorted(df[case_col].dropna().unique()))
    if len(cases) == 0:
        return float("nan"), float("nan"), float("nan")
    values: list[float] = []
    for _ in range(n):
        sampled = np.random.choice(cases, size=len(cases), replace=True)
        pieces = [df[df[case_col] == case] for case in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        value = reducer(boot)
        if not math.isnan(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def rg_context(pattern: str) -> str:
    try:
        proc = subprocess.run(
            [
                "rg",
                "-n",
                "-C",
                "2",
                "--fixed-strings",
                "--glob",
                "!phase2/final_data/canonical/**",
                pattern,
                ".",
            ],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"RG_FAILED: {exc}"


def rg_context_markdown(pattern: str) -> str:
    try:
        proc = subprocess.run(
            [
                "rg",
                "-n",
                "-C",
                "2",
                "--fixed-strings",
                "--glob",
                "*.md",
                "--glob",
                "!phase2/final_data/canonical/**",
                pattern,
                ".",
            ],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"RG_FAILED: {exc}"


def task1_topology() -> dict[str, Any]:
    print("\nTASK 1 topology_auc")
    missing: list[str] = []
    candidates = [
        ROOT / "phase2" / "_archive" / "p3_cubical_persistence" / "p3_component_persistence.csv",
        ROOT / "phase2" / "p3_cubical_persistence" / "p3_component_persistence.csv",
    ]
    source_file = next((p for p in candidates if p.exists()), None)
    if source_file is None:
        missing.extend(str(p.relative_to(ROOT)) for p in candidates)
        print("MISSING " + "; ".join(missing))
        result = {
            "A_naive": None,
            "A_corr": None,
            "A_corr_ci95": [None, None],
            "n_ET_comp": None,
            "n_conf_comp": None,
            "n_cases_eff": None,
            "case_level_auc": None,
            "source_file": None,
            "notes_0p714": "missing component table",
        }
        (OUT / "topology_auc.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    comp = pd.read_csv(source_file)
    used_cols = ["case_id", "source", "ph_h0_max_persistence"]
    print(f"source={source_file.relative_to(ROOT)}")
    print("columns_used=" + ",".join(used_cols))

    comp["score"] = numeric(comp["ph_h0_max_persistence"])
    et = comp[comp["source"].isin(["true_et_cohort", "true_et_old20"]) & comp["score"].notna()].copy()
    conf = comp[
        comp["source"].isin(["absent_fp_component", "peri_cavity_fp_component"]) & comp["score"].notna()
    ].copy()
    pooled = pd.concat(
        [
            et.assign(label_et=1, label_conf=0),
            conf.assign(label_et=0, label_conf=1),
        ],
        ignore_index=True,
    )
    a_naive = auc_binary(pooled["label_et"].to_numpy(), pooled["score"].to_numpy())
    a_corr = auc_binary(pooled["label_conf"].to_numpy(), pooled["score"].to_numpy())
    complement_error = abs(a_corr - (1.0 - a_naive))

    cases_eff = int(pooled["case_id"].nunique())

    cases = np.array(sorted(pooled["case_id"].dropna().unique()))
    labels_arr = pooled["label_conf"].to_numpy(dtype=int)
    scores_arr = pooled["score"].to_numpy(dtype=float)
    case_to_indices = {
        case: pooled.index[pooled["case_id"] == case].to_numpy(dtype=int)
        for case in cases
    }
    boot_values: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = np.random.choice(cases, size=len(cases), replace=True)
        idx = np.concatenate([case_to_indices[case] for case in sampled])
        if len(np.unique(labels_arr[idx])) < 2:
            continue
        boot_values.append(auc_binary(labels_arr[idx], scores_arr[idx]))
    if boot_values:
        ci_low = float(np.percentile(boot_values, 2.5))
        ci_high = float(np.percentile(boot_values, 97.5))
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    case_level = (
        pooled.groupby(["case_id", "label_conf"], as_index=False)["score"]
        .max()
        .rename(columns={"score": "case_max_score"})
    )
    case_level_auc = auc_binary(case_level["label_conf"].to_numpy(), case_level["case_max_score"].to_numpy())

    matches_0714 = rg_context("0." + "714")
    matches_0874 = rg_context_markdown("0." + "874")
    recomputed_3 = f"{a_corr:.3f}"
    recomputed_4 = f"{a_corr:.4f}"
    matches_recomputed = "\n".join(
        x for x in [matches_0874, rg_context_markdown(recomputed_3), rg_context_markdown(recomputed_4)] if x
    ).strip()
    notes = {
        "literal_0.714_matches": matches_0714 or "NONE",
        "literal_0.874_or_recomputed_matches": matches_recomputed or "NONE",
        "complement_error": complement_error,
    }

    result = {
        "A_naive": a_naive,
        "A_corr": a_corr,
        "A_corr_ci95": [ci_low, ci_high],
        "n_ET_comp": int(len(et)),
        "n_conf_comp": int(len(conf)),
        "n_cases_eff": cases_eff,
        "case_level_auc": case_level_auc,
        "source_file": str(source_file.relative_to(ROOT)),
        "columns_used": used_cols,
        "notes_0p714": notes,
    }
    (OUT / "topology_auc.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"A_naive={a_naive:.6f} A_corr={a_corr:.6f} "
        f"CI95=[{ci_low:.6f},{ci_high:.6f}] n_ET={len(et)} n_conf={len(conf)} "
        f"n_cases_eff={cases_eff} complement_error={complement_error:.3g}"
    )
    print("grep_0p714:")
    print(matches_0714 or "NONE")
    print("grep_0p874_or_recomputed_in_markdown:")
    print(matches_recomputed or "NONE")
    return result


def representative_spacing(manifest: pd.DataFrame, stage4: pd.DataFrame | None) -> tuple[float, str]:
    search_roots = [ROOT / "images", ROOT / "BraTS2024-BraTS-GLI-TrainingData" / "training_data1_v2"]
    for case_id in manifest["case_id"].astype(str):
        for base in search_roots:
            seg = base / case_id / f"{case_id}-seg.nii.gz"
            if seg.exists():
                try:
                    import SimpleITK as sitk

                    img = sitk.ReadImage(str(seg))
                    spacing = img.GetSpacing()
                    vox_mm3 = float(spacing[0] * spacing[1] * spacing[2])
                    return vox_mm3, f"NIfTI spacing from {seg.relative_to(ROOT)}"
                except Exception as exc:
                    return float("nan"), f"failed reading {seg.relative_to(ROOT)}: {exc}"
    if stage4 is not None and {"et_mm3_manifest", "gt_vox"}.issubset(stage4.columns):
        present = stage4[numeric(stage4["gt_vox"]) > 0].copy()
        ratios = numeric(present["et_mm3_manifest"]) / numeric(present["gt_vox"])
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
        if not ratios.empty:
            return float(ratios.median()), "median(et_mm3_manifest / gt_vox) from stage4_case_metrics.csv"
    return float("nan"), "MISSING representative NIfTI and voxel-volume fallback"


def parse_cutoffs() -> tuple[float | None, float | None, str]:
    spec = ROOT / "cohort" / "COHORT_MANIFEST.md"
    if not spec.exists():
        return None, None, "MISSING cohort/COHORT_MANIFEST.md"
    text = spec.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"p33/p66\s*=\s*`?([0-9.]+)`?\s*/\s*`?([0-9.]+)`?\s*mm", text)
    if not match:
        return None, None, "cutoffs not found in cohort/COHORT_MANIFEST.md"
    return float(match.group(1)), float(match.group(2)), "cohort/COHORT_MANIFEST.md p33/p66"


def task2_units() -> dict[str, Any]:
    print("\nTASK 2 physical_units")
    missing: list[str] = []
    manifest = read_csv(ROOT / "phase2" / "final_data" / "cohort_manifest_selected.csv", missing)
    stage4 = read_csv(ROOT / "phase2" / "final_data" / "stage4_case_metrics.csv", missing)
    rows: list[dict[str, Any]] = []
    if manifest is None:
        print("MISSING " + "; ".join(missing))
        pd.DataFrame(rows).to_csv(OUT / "units.csv", index=False)
        return {"voxel_mm3": None, "flood_mL": None, "cutoffs_mL": {}}

    vox_mm3, vox_source = representative_spacing(manifest, stage4)
    flood_vox = DEFAULT_CONFIG.flood_threshold_vox
    rows.append(
        {
            "quantity": "voxel_volume",
            "value_voxels": 1,
            "value_mL": vox_mm3 / 1000.0 if not math.isnan(vox_mm3) else np.nan,
            "source": vox_source,
        }
    )
    rows.append(
        {
            "quantity": "flood_threshold",
            "value_voxels": flood_vox,
            "value_mL": flood_vox * vox_mm3 / 1000.0 if not math.isnan(vox_mm3) else np.nan,
            "source": f"phase2.metrics.DEFAULT_CONFIG.flood_threshold_vox; {vox_source}",
        }
    )
    p33, p66, cutoff_source = parse_cutoffs()
    cutoffs_mL: dict[str, float | None] = {}
    if p33 is not None and p66 is not None and not math.isnan(vox_mm3):
        rows.extend(
            [
                {
                    "quantity": "small_upper_cutoff_p33",
                    "value_voxels": p33 / vox_mm3,
                    "value_mL": p33 / 1000.0,
                    "source": cutoff_source,
                },
                {
                    "quantity": "large_lower_cutoff_p66",
                    "value_voxels": p66 / vox_mm3,
                    "value_mL": p66 / 1000.0,
                    "source": cutoff_source,
                },
            ]
        )
        cutoffs_mL = {"small_upper_p33": p33 / 1000.0, "large_lower_p66": p66 / 1000.0}
    else:
        process = manifest[pd.to_numeric(manifest["process"], errors="coerce") == 1].copy()
        for vol_bin, group in process.groupby("vol_bin"):
            vals = numeric(group["et_mm3"]).dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "quantity": f"{vol_bin}_empirical_min_median_max",
                    "value_voxels": f"{vals.min() / vox_mm3},{vals.median() / vox_mm3},{vals.max() / vox_mm3}",
                    "value_mL": f"{vals.min() / 1000.0},{vals.median() / 1000.0},{vals.max() / 1000.0}",
                    "source": "cohort_manifest_selected.csv empirical process cohort",
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "units.csv", index=False)
    flood_ml = flood_vox * vox_mm3 / 1000.0 if not math.isnan(vox_mm3) else float("nan")
    print(f"voxel_mm3={vox_mm3:.6f} flood_threshold_mL={flood_ml:.6f} source={vox_source}")
    if cutoffs_mL:
        print(f"cutoffs_mL={cutoffs_mL}")
    else:
        print("cutoffs missing; wrote empirical stratum summaries")
    return {"voxel_mm3": vox_mm3, "flood_mL": flood_ml, "cutoffs_mL": cutoffs_mL}


def headline_aggregates(stage4: pd.DataFrame, cases: set[str] | None = None) -> pd.DataFrame:
    df = stage4.copy()
    if cases is not None:
        df = df[df["case_id"].isin(cases)].copy()
    rows: list[dict[str, Any]] = []
    for method, g in df.groupby("metodo"):
        absent = g[g["vol_bin"] == "absent"].copy()
        large = g[g["vol_bin"] == "large"].copy()
        rows.extend(
            [
                {
                    "method": method,
                    "metric": "correct_absent_rate",
                    "value": numeric(absent["correct_absent_pred_lt_10_vox"]).mean(),
                },
                {
                    "method": method,
                    "metric": "flood_rate",
                    "value": numeric(absent["flood_gt_10000_vox"]).mean(),
                },
                {
                    "method": method,
                    "metric": "median_FP_volume_voxels",
                    "value": numeric(absent["pred_vox"]).median(),
                },
                {
                    "method": method,
                    "metric": "large_stratum_lesionwise_dice",
                    "value": numeric(large["lesionwise_dice_mean"]).mean(),
                },
            ]
        )
    return pd.DataFrame(rows)


def task3_leakage() -> tuple[pd.DataFrame, dict[str, str]]:
    print("\nTASK 3 leakage_sensitivity")
    missing: list[str] = []
    manifest = read_csv(ROOT / "phase2" / "final_data" / "cohort_manifest_selected.csv", missing)
    stage4 = read_csv(ROOT / "phase2" / "final_data" / "stage4_case_metrics.csv", missing)
    if manifest is None or stage4 is None:
        print("MISSING " + "; ".join(missing))
        out = pd.DataFrame(columns=["method", "metric", "full", "minus_overlap", "delta"])
        out.to_csv(OUT / "leakage_sensitivity.csv", index=False)
        return out, {}

    intended = ["BraTS-GLI-02086-100", "BraTS-GLI-02143-100", "BraTS-GLI-02151-100"]
    process = manifest[pd.to_numeric(manifest["process"], errors="coerce") == 1].copy()
    overlap = {
        case_id: ("present" if case_id in set(process["case_id"].astype(str)) else "MISSING")
        for case_id in intended
    }
    ambiguous_02143_hits = process[
        process["case_id"].astype(str).str.contains("02143", regex=False)
    ]["case_id"].astype(str).tolist()
    overlap_cases = {case_id for case_id, status in overlap.items() if status == "present"}
    full_cases = set(process["case_id"].astype(str))
    minus_cases = full_cases - overlap_cases

    full = headline_aggregates(stage4, full_cases).rename(columns={"value": "full"})
    minus = headline_aggregates(stage4, minus_cases).rename(columns={"value": "minus_overlap"})
    out = full.merge(minus, on=["method", "metric"], how="outer")
    out["delta"] = out["minus_overlap"] - out["full"]
    out.to_csv(OUT / "leakage_sensitivity.csv", index=False)
    print(f"overlap_confirmed={overlap} ambiguous_02143_hits={ambiguous_02143_hits}")
    for method in ["variational_spline", "gmm_2d"]:
        sub = out[out["method"] == method]
        print(method + " " + json.dumps(dict(zip(sub["metric"], sub["delta"])), sort_keys=True))
    return out, overlap


def task4_baseline_cis(units: dict[str, Any]) -> pd.DataFrame:
    print("\nTASK 4 baseline_case_clustered_cis")
    missing: list[str] = []
    stage4 = read_csv(ROOT / "phase2" / "final_data" / "stage4_case_metrics.csv", missing)
    if stage4 is None:
        print("MISSING " + "; ".join(missing))
        out = pd.DataFrame()
        out.to_csv(OUT / "baseline_headline_cis.csv", index=False)
        return out

    rows: list[dict[str, Any]] = []
    def array_bootstrap(values: np.ndarray, reducer) -> tuple[float, float, float, float]:
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        point = float(reducer(values))
        boots = np.empty(BOOTSTRAPS, dtype=float)
        for i in range(BOOTSTRAPS):
            sample = np.random.choice(values, size=len(values), replace=True)
            boots[i] = float(reducer(sample))
        return (
            point,
            float(np.mean(boots)),
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        )

    for method, g in stage4.groupby("metodo"):
        absent = g[g["vol_bin"] == "absent"].copy()
        large = g[g["vol_bin"] == "large"].copy()
        metric_specs = [
            ("flood_rate", numeric(absent["flood_gt_10000_vox"]).to_numpy(dtype=float), np.mean, absent),
            ("median_FP_volume", numeric(absent["pred_vox"]).to_numpy(dtype=float), np.median, absent),
            (
                "large_stratum_lesionwise_dice",
                numeric(large["lesionwise_dice_mean"]).to_numpy(dtype=float),
                np.mean,
                large,
            ),
        ]
        for metric, values, reducer, df_metric in metric_specs:
            point, mean, lo, hi = array_bootstrap(values, reducer)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "point_estimate": point,
                    "bootstrap_mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_cases": int(df_metric["case_id"].nunique()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "baseline_headline_cis.csv", index=False)
    print(f"wrote baseline_headline_cis.csv rows={len(out)}")
    return out


def first_numeric(*values: Any) -> float | None:
    for value in values:
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(num):
            return float(num)
    return None


def task5_operating_points(units: dict[str, Any], baseline_cis: pd.DataFrame) -> pd.DataFrame:
    print("\nTASK 5 reconcile_master_growth")
    missing: list[str] = []
    master = read_csv(ROOT / "phase2" / "final_data" / "four_regime_master_comparison.csv", missing)
    growth = read_csv(ROOT / "phase2" / "final_data" / "growth_metric_table.csv", missing)
    p2b = read_csv(ROOT / "phase2" / "final_data" / "p2_soft_shape_operating_points.csv", missing)
    p3 = read_csv(ROOT / "phase2" / "final_data" / "p3_key_comparison_vs_baseline.csv", missing)
    if master is None:
        print("MISSING " + "; ".join(missing))
        out = pd.DataFrame()
        out.to_csv(OUT / "canonical_operating_points.csv", index=False)
        return out

    vox_ml = float(units.get("voxel_mm3", np.nan)) / 1000.0
    rows: list[dict[str, Any]] = []
    conflicts: list[str] = []
    regime_status = {"R1": "locked", "R2": "post_hoc", "R3": "post_hoc", "R4": "exploratory"}
    regime_names = {
        "R1": "INTENSITY_BASELINE",
        "R2": "SPATIAL_LOCATION_PRIOR",
        "R3": "SHAPE_PROXY_GEOMETRY",
        "R4": "PERSISTENT_HOMOLOGY",
    }

    for _, row in master.iterrows():
        method = row["method"]
        for regime in ["R1", "R2", "R3", "R4"]:
            setting = row.get(f"{regime}_operating_point", "")
            m_flood = row.get(f"{regime}_absent_flood_rate")
            m_fp = row.get(f"{regime}_absent_median_fp_vox")
            m_dice = row.get(f"{regime}_large_lesionwise_dice")
            m_valid = str(row.get(f"{regime}_valid_operating_point", "")).lower() == "true"

            source_note = "four_regime_master_comparison.csv"
            flood = first_numeric(m_flood)
            fp_vox = first_numeric(m_fp)
            dice = first_numeric(m_dice)

            if regime == "R3" and p2b is not None:
                p2_rows = p2b[p2b["method"] == method]
                if not p2_rows.empty and (flood is None or fp_vox is None or dice is None):
                    p2_row = p2_rows.iloc[-1]
                    conflicts.append(
                        f"R3 {method}: master empty/invalid, p2_soft_shape_operating_points.csv has numeric threshold {p2_row.get('shape_score_threshold')}"
                    )
                    flood = first_numeric(p2_row.get("absent_flood_rate"))
                    fp_vox = first_numeric(p2_row.get("absent_median_fp_vox"))
                    dice = first_numeric(p2_row.get("large_lesionwise_mean"))
                    setting = f"shape_score_threshold@{p2_row.get('shape_score_threshold')}"
                    source_note += "; filled from p2_soft_shape_operating_points.csv"

            if regime == "R4":
                p3_row = None
                if p3 is not None and not p3[p3["method"] == method].empty:
                    p3_row = p3[p3["method"] == method].iloc[0]
                growth_row = None
                if growth is not None and not growth[growth["method"] == method].empty:
                    growth_row = growth[growth["method"] == method].iloc[0]
                p3_flood = first_numeric(None if p3_row is None else p3_row.get("absent_flood_rate"))
                growth_flood = first_numeric(
                    None if growth_row is None else growth_row.get("p3_cubical_persistence_absent_flood_rate")
                )
                if flood is None and (p3_flood is not None or growth_flood is not None):
                    conflicts.append(
                        f"R4 {method}: master empty/invalid, p3_key/growth CSV has numeric held-out P3 values"
                    )
                if p3_flood is not None:
                    flood = p3_flood
                    fp_vox = first_numeric(p3_row.get("absent_median_fp_vox"))
                    dice = first_numeric(p3_row.get("large_lesionwise_mean"))
                    source_note += "; reconciled from p3_key_comparison_vs_baseline.csv"
                elif growth_flood is not None:
                    flood = growth_flood
                    fp_vox = first_numeric(growth_row.get("p3_cubical_persistence_absent_median_fp_vox"))
                    dice = first_numeric(growth_row.get("p3_cubical_persistence_large_lesionwise_dice"))
                    source_note += "; reconciled from growth_metric_table.csv"

            valid_by_rule = (fp_vox is not None and fp_vox > 0) and (dice is not None or flood is not None)
            status = regime_status[regime] if valid_by_rule else "invalid"
            if not m_valid and valid_by_rule:
                source_note += "; master valid flag false but canonical validity rule passes"
            if m_valid and not valid_by_rule:
                source_note += "; master valid flag true but canonical validity rule fails"

            row_out: dict[str, Any] = {
                "regime": f"{regime}_{regime_names[regime]}",
                "method": method,
                "setting": setting,
                "status": status,
                "flood_rate": flood,
                "median_FP_mL": fp_vox * vox_ml if fp_vox is not None and not math.isnan(vox_ml) else np.nan,
                "large_dice": dice,
                "notes": source_note,
            }
            if regime == "R1" and not baseline_cis.empty:
                cis = baseline_cis[baseline_cis["method"] == method]
                for metric_name, prefix in [
                    ("flood_rate", "flood_rate"),
                    ("median_FP_volume", "median_FP_volume"),
                    ("large_stratum_lesionwise_dice", "large_dice"),
                ]:
                    c = cis[cis["metric"] == metric_name]
                    if not c.empty:
                        row_out[f"{prefix}_ci95_low"] = c.iloc[0]["ci95_low"]
                        row_out[f"{prefix}_ci95_high"] = c.iloc[0]["ci95_high"]
            rows.append(row_out)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "canonical_operating_points.csv", index=False)
    if conflicts:
        print("conflicts:")
        for conflict in conflicts:
            print("- " + conflict)
    else:
        print("conflicts: NONE")
    print(f"wrote canonical_operating_points.csv rows={len(out)}")
    return out


def task6_surface_rho() -> float:
    print("\nTASK 6 surface_correlation")
    missing: list[str] = []
    surf = read_csv(ROOT / "phase2" / "final_data" / "surface_prediction_fidelity_by_method_stratum.csv", missing)
    if surf is None:
        print("MISSING " + "; ".join(missing))
        return float("nan")
    large = surf[surf["vol_bin"] == "large"].copy()
    from scipy.stats import spearmanr

    rho = float(
        spearmanr(
            numeric(large["lesionwise_dice_mean"]).to_numpy(),
            numeric(large["surface_asd_median"]).to_numpy(),
        ).statistic
    )
    print(f"large_stratum_spearman_lesionwise_vs_surface_ASD_rho={rho:.6f} n={len(large)} descriptive_only")
    return rho


def format_delta(leakage: pd.DataFrame, method: str) -> str:
    if leakage.empty:
        return f"{method}: MISSING"
    sub = leakage[leakage["method"] == method]
    parts = []
    for metric in ["flood_rate", "median_FP_volume_voxels", "large_stratum_lesionwise_dice"]:
        row = sub[sub["metric"] == metric]
        if not row.empty:
            parts.append(f"{metric} delta={row.iloc[0]['delta']:.6g}")
    return f"{method}: " + ", ".join(parts)


def paste_back(
    topo: dict[str, Any],
    units: dict[str, Any],
    leakage: pd.DataFrame,
    ops: pd.DataFrame,
    surface_rho: float,
) -> None:
    r3_vs = ops[(ops["regime"].str.startswith("R3_")) & (ops["method"] == "variational_spline")]
    r1_vs = ops[(ops["regime"].str.startswith("R1_")) & (ops["method"] == "variational_spline")]
    r3_line = "R3 variational_spline: MISSING"
    if not r3_vs.empty and not r1_vs.empty:
        r1 = r1_vs.iloc[0]
        r3 = r3_vs.iloc[0]
        r3_line = (
            "R3 variational_spline tradeoff: "
            f"baseline flood={r1['flood_rate']:.6f}, FP={r1['median_FP_mL']:.3f} mL, Dice={r1['large_dice']:.6f} "
            f"-> {r3['setting']} flood={r3['flood_rate']:.6f}, FP={r3['median_FP_mL']:.3f} mL, Dice={r3['large_dice']:.6f}"
        )

    cutoffs = units.get("cutoffs_mL") or {}
    lines = [
        "```PASTE_BACK",
        f"A_naive_ET_above_confound={topo.get('A_naive'):.6f}",
        f"A_corr_confound_above_ET={topo.get('A_corr'):.6f} CI95=[{topo.get('A_corr_ci95')[0]:.6f},{topo.get('A_corr_ci95')[1]:.6f}]",
        f"PH components: ET={topo.get('n_ET_comp')} confound={topo.get('n_conf_comp')} n_cases_eff={topo.get('n_cases_eff')}",
        f"case_level_AUC_confound_above_ET={topo.get('case_level_auc'):.6f}",
        f"flood_threshold={DEFAULT_CONFIG.flood_threshold_vox} vox = {units.get('flood_mL'):.3f} mL",
        f"size cutoffs: small<p33 {cutoffs.get('small_upper_p33', float('nan')):.3f} mL; large>=p66 {cutoffs.get('large_lower_p66', float('nan')):.3f} mL",
        format_delta(leakage, "variational_spline"),
        format_delta(leakage, "gmm_2d"),
        r3_line,
        f"surface rho large lesionwise Dice vs ASD={surface_rho:.6f} (n=5, descriptive only)",
        "```",
    ]
    print("\n".join(lines))


def main() -> None:
    print_versions()
    topo = task1_topology()
    units = task2_units()
    leakage, _ = task3_leakage()
    baseline_cis = task4_baseline_cis(units)
    ops = task5_operating_points(units, baseline_cis)
    surface_rho = task6_surface_rho()
    paste_back(topo, units, leakage, ops, surface_rho)


if __name__ == "__main__":
    main()
