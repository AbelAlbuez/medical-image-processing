from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.segmentation import morphological_chan_vese


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline import config, io_utils  # noqa: E402
from brats_pipeline.seg_et_pipeline import correr_pipeline_et  # noqa: E402
from brats_pipeline.seg_spline_levelset import (  # noqa: E402
    _bbox,
    _chanvese_iterate_score,
    _orientar_chanvese,
    _reset_morphsnakes_curvop,
    roi_et_auto,
)


BASELINE = ROOT / "analysis" / "baseline" / "metricas_ET_baseline.csv"
OFF_EQUIV = ROOT / "analysis" / "stage3b_off_equivalence.csv"
OUT_VS = ROOT / "analysis" / "stage3b_variational_spline.csv"
OUT_SACRED = ROOT / "analysis" / "stage3b_sacred14.csv"
OUT_TRAJ = ROOT / "analysis" / "stage3b_trajectory_scores.csv"
TARGETS = {"BraTS-GLI-00506-100", "BraTS-GLI-02139-100", "BraTS-GLI-02169-100"}


def read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def clean(case_id: str, mod: str):
    return read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")


def raw(case_id: str, mod: str):
    return read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")


def gt(case_id: str):
    return sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz"))


def mask_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr.astype(np.uint8)).tobytes()).hexdigest()


def run_case(case_id: str):
    masks, _, df = correr_pipeline_et(
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
    return masks, df


def mode_off() -> None:
    baseline = pd.read_csv(BASELINE)
    rows = []
    for case_id in sorted(baseline["case_id"].unique()):
        masks, _ = run_case(case_id)
        observed = masks["variational_spline"]
        disk = sitk.GetArrayFromImage(
            sitk.ReadImage(str(ROOT / "output" / "segmentacion" / case_id / f"{case_id}-et_variational_spline.nii.gz"))
        )
        rows.append(
            {
                "case_id": case_id,
                "observed_hash": mask_hash(observed),
                "frozen_mask_hash": mask_hash(disk),
                "bit_identical": mask_hash(observed) == mask_hash(disk),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OFF_EQUIV, index=False)
    print(out.to_string(index=False))
    if not out["bit_identical"].all():
        raise SystemExit("OFF equivalence failed")


def trajectory_scores(case_id: str, selected_iter: int | None) -> pd.DataFrame:
    t1c = clean(case_id, "t1c")
    arr_t1c = io_utils.a_numpy(t1c).astype(np.float32)
    arr_t1c_raw = io_utils.a_numpy(raw(case_id, "t1c")).astype(np.float32)
    arr_t1n_raw = io_utils.a_numpy(raw(case_id, "t1n")).astype(np.float32)
    roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5)
    sl = _bbox(roi, margin=12, shape=mapa.shape)
    img = mapa[sl].astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    init = roi[sl].astype(np.uint8)
    full_init = np.zeros_like(mapa, dtype=np.uint8)
    full_init[sl] = init
    snapshots = {}
    callback_count = {"i": 0}

    def callback(u):
        snapshots[callback_count["i"]] = np.array(u, copy=True)
        callback_count["i"] += 1

    _reset_morphsnakes_curvop("first")
    morphological_chan_vese(
        img,
        num_iter=35,
        init_level_set=init,
        smoothing=3,
        lambda1=1.0,
        lambda2=1.0,
        iter_callback=callback,
    )
    rows = []
    prev_voxels = None
    best_score = -np.inf
    drops = 0
    selected_by_replay = None
    for iteration in sorted(snapshots):
        ls = _orientar_chanvese(snapshots[iteration].astype(np.uint8), img)
        pred = np.zeros_like(mapa, dtype=np.uint8)
        pred[sl] = ls
        acceptable, score, info = _chanvese_iterate_score(
            pred, init=full_init, mapa=mapa, prev_voxels=prev_voxels
        )
        voxels = int(pred.sum())
        prev_voxels = voxels
        if iteration > 0 and acceptable and score > best_score:
            selected_by_replay = iteration
            best_score = score
            drops = 0
        elif iteration > 0 and best_score > -np.inf and score < best_score:
            drops += 1
            if drops >= config.BEST_ITERATE_PATIENCE:
                rows.append(
                    {
                        "case_id": case_id,
                        "iter": iteration,
                        "voxels": voxels,
                        "acceptable": acceptable,
                        "score": score,
                        "selected_iter": selected_iter,
                        "selected_by_replay": selected_by_replay,
                        "patience_stop": True,
                        **info,
                    }
                )
                break
        else:
            drops = 0
        rows.append(
            {
                "case_id": case_id,
                "iter": iteration,
                "voxels": voxels,
                "acceptable": acceptable,
                "score": score,
                "selected_iter": selected_iter,
                "selected_by_replay": selected_by_replay,
                "patience_stop": False,
                **info,
            }
        )
    return pd.DataFrame(rows)


def mode_on() -> None:
    baseline = pd.read_csv(BASELINE)
    current = []
    traj = []
    for case_id in sorted(baseline["case_id"].unique()):
        _, df = run_case(case_id)
        current.append(df)
        row = df[df["metodo"] == "variational_spline"].iloc[0]
        if case_id in TARGETS:
            selected_iter = int(row.get("best_iter")) if "best_iter" in row and pd.notna(row.get("best_iter")) else None
            traj.append(trajectory_scores(case_id, selected_iter))
    current = pd.concat(current, ignore_index=True)
    base_vs = baseline[baseline["metodo"] == "variational_spline"]
    new_vs = current[current["metodo"] == "variational_spline"]
    compare = base_vs.merge(new_vs, on=["case_id", "metodo"], suffixes=("_baseline", "_stage3b"))
    compare["dice_delta"] = compare["dice_ET_stage3b"] - compare["dice_ET_baseline"]
    compare["was_clean_evolved"] = compare["guard_branch_baseline"] == "evolved"
    compare["clean_case_regressed"] = compare["was_clean_evolved"] & (compare["dice_delta"] < -1e-9)
    compare["guard_rejects_returned_iter"] = compare["guard_branch_stage3b"] == "collapse-detected"

    cols = [
        "case_id",
        "dice_ET_baseline",
        "dice_ET_stage3b",
        "dice_delta",
        "guard_branch_baseline",
        "guard_reason_baseline",
        "guard_branch_stage3b",
        "guard_reason_stage3b",
        "was_clean_evolved",
        "clean_case_regressed",
        "guard_rejects_returned_iter",
    ]
    for optional in ["best_iter", "best_score", "best_acceptable", "best_voxels"]:
        col = f"{optional}_stage3b"
        if col in compare.columns:
            cols.append(col)
    compare[cols].to_csv(OUT_VS, index=False)
    compare[compare["was_clean_evolved"]][cols].to_csv(OUT_SACRED, index=False)
    if traj:
        pd.concat(traj, ignore_index=True).to_csv(OUT_TRAJ, index=False)

    print("variational_spline")
    print(compare[cols].round(4).to_string(index=False))
    print("\nsacred regressions")
    print(compare[compare["clean_case_regressed"]][cols].to_string(index=False))
    print("\nguard rejects returned iter")
    print(compare[compare["guard_rejects_returned_iter"]][cols].to_string(index=False))
    if compare["clean_case_regressed"].any():
        raise SystemExit("Sacred-14 gate failed")
    if compare["guard_rejects_returned_iter"].any():
        raise SystemExit("3A consistency gate failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["off", "on"], required=True)
    args = parser.parse_args()
    if args.mode == "off":
        mode_off()
    else:
        mode_on()


if __name__ == "__main__":
    main()
