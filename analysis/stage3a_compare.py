from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline.seg_et_pipeline import correr_pipeline_et  # noqa: E402


BASELINE = ROOT / "analysis" / "baseline" / "metricas_ET_baseline.csv"
OUT_CASES = ROOT / "analysis" / "stage3a_variational_spline.csv"
OUT_GUARD = ROOT / "analysis" / "stage3a_guard_firing_log.csv"
OUT_GUARD_DIFF = ROOT / "analysis" / "stage3a_guard_firing_log_diff.csv"
OUT_GUARD_CASE_DIFF = ROOT / "analysis" / "stage3a_guard_case_diff.csv"


def read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def clean(case_id: str, mod: str):
    return read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")


def raw(case_id: str, mod: str):
    return read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")


def gt(case_id: str):
    return sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz"))


def run_case(case_id: str) -> pd.DataFrame:
    _, _, df = correr_pipeline_et(
        clean(case_id, "t1c"),
        clean(case_id, "t1n"),
        gt(case_id),
        t1c_raw=raw(case_id, "t1c"),
        t1n_raw=raw(case_id, "t1n"),
        t2f=clean(case_id, "t2f"),
        semilla_zyx=None,
        case_id=case_id,
        auto_pct=90.0,
        sigma=0.5,
        verbose=False,
    )
    return df


def summarize_guard(df: pd.DataFrame) -> pd.DataFrame:
    deformable = df[df["guard_branch"].fillna("") != ""].copy()
    deformable = deformable.rename(columns={"metodo": "method"})
    evidence_cols = [
        "evidence_lcc_fraction",
        "evidence_pred_enhancement",
        "evidence_init_enhancement",
        "evidence_enhancement_ratio",
        "evidence_volume_multiple",
        "evidence_accept",
    ]
    for col in evidence_cols:
        if col not in deformable.columns:
            deformable[col] = ""
    return deformable[[
        "case_id",
        "method",
        "dice_ET",
        "guard_branch",
        "guard_reason",
        "vol_pred",
        "evidence_lcc_fraction",
        "evidence_pred_enhancement",
        "evidence_init_enhancement",
        "evidence_enhancement_ratio",
        "evidence_volume_multiple",
        "evidence_accept",
    ]]


def grouped_guard(df: pd.DataFrame, label: str) -> pd.DataFrame:
    g = (
        df.groupby(["method", "guard_branch", "guard_reason"], dropna=False)
        .size()
        .reset_index(name=label)
    )
    return g


def main() -> None:
    baseline = pd.read_csv(BASELINE)
    cases = sorted(baseline["case_id"].unique())
    current = pd.concat([run_case(case) for case in cases], ignore_index=True)

    base_vs = baseline[baseline["metodo"] == "variational_spline"].copy()
    new_vs = current[current["metodo"] == "variational_spline"].copy()
    compare = base_vs.merge(
        new_vs,
        on=["case_id", "metodo"],
        suffixes=("_baseline", "_stage3a"),
    )
    compare["dice_delta"] = compare["dice_ET_stage3a"] - compare["dice_ET_baseline"]
    compare["was_clean_evolved"] = compare["guard_branch_baseline"] == "evolved"
    compare["clean_case_regressed"] = (
        compare["was_clean_evolved"] & (compare["dice_delta"] < -1e-9)
    )
    cols = [
        "case_id",
        "dice_ET_baseline",
        "dice_ET_stage3a",
        "dice_delta",
        "guard_branch_baseline",
        "guard_reason_baseline",
        "guard_branch_stage3a",
        "guard_reason_stage3a",
        "was_clean_evolved",
        "clean_case_regressed",
    ]
    compare[cols].to_csv(OUT_CASES, index=False)

    guard = summarize_guard(current)
    guard.to_csv(OUT_GUARD, index=False)

    base_guard = summarize_guard(baseline)
    base_group = grouped_guard(base_guard, "baseline_n")
    new_group = grouped_guard(guard, "stage3a_n")
    diff = base_group.merge(
        new_group,
        on=["method", "guard_branch", "guard_reason"],
        how="outer",
    ).fillna(0)
    diff["delta_n"] = diff["stage3a_n"].astype(int) - diff["baseline_n"].astype(int)
    diff = diff.sort_values(["method", "guard_branch", "guard_reason"])
    diff.to_csv(OUT_GUARD_DIFF, index=False)

    guard_case = base_guard.merge(
        guard,
        on=["case_id", "method"],
        suffixes=("_baseline", "_stage3a"),
    )
    changed = guard_case[
        (guard_case["guard_branch_baseline"] != guard_case["guard_branch_stage3a"])
        | (guard_case["guard_reason_baseline"] != guard_case["guard_reason_stage3a"])
    ]
    changed.to_csv(OUT_GUARD_CASE_DIFF, index=False)

    print("variational_spline")
    print(compare[cols].round(4).to_string(index=False))
    print("\nmeans")
    print(
        compare[["dice_ET_baseline", "dice_ET_stage3a"]]
        .mean()
        .rename({"dice_ET_baseline": "baseline", "dice_ET_stage3a": "stage3a"})
        .round(4)
        .to_string()
    )
    print("\nclean evolved regressions")
    print(compare[compare["clean_case_regressed"]][cols].to_string(index=False))
    print("\nguard diff")
    print(diff.to_string(index=False))
    print("\nguard case changes")
    print(changed.to_string(index=False))


if __name__ == "__main__":
    main()
