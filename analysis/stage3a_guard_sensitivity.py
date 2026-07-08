from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage.segmentation import morphological_chan_vese


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline import config, io_utils  # noqa: E402
from brats_pipeline.seg_metrics import dice  # noqa: E402
from brats_pipeline.seg_spline_levelset import (  # noqa: E402
    _bbox,
    _morfologia,
    _orientar_chanvese,
    _reset_morphsnakes_curvop,
    _restringir_realce,
    _semilla_degenerada,
    roi_et_auto,
)


BASELINE = ROOT / "analysis" / "baseline" / "metricas_ET_baseline.csv"
OUT = ROOT / "analysis" / "stage3a_guard_sensitivity.csv"
ST = ndimage.generate_binary_structure(3, 1)
ENH_GRID = [0.80, 0.85, 0.90, 0.95, 1.00]
LCC_GRID = [0.80, 0.85, 0.90, 0.95]
VOL_GRID = [1.50, 2.00, 2.50, 3.00]
FIXED_ENH = 0.90
FIXED_LCC = 0.90
FIXED_VOL = 2.50


def read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def clean(case_id: str, mod: str):
    return read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")


def raw(case_id: str, mod: str):
    return read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")


def gt_et(case_id: str, ref: sitk.Image) -> np.ndarray:
    seg = sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz"))
    seg = sitk.Resample(
        sitk.Cast(seg, sitk.sitkInt16),
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkInt16,
    )
    return (sitk.GetArrayFromImage(seg) == config.LABEL_ET).astype(np.uint8)


def lcc_fraction(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    lab, n = ndimage.label(mask > 0, structure=ST)
    if n == 0:
        return 0.0
    sizes = ndimage.sum(mask > 0, lab, range(1, n + 1))
    return float(np.max(sizes) / float(mask.sum()))


def raw_variational_candidate(case_id: str) -> dict:
    t1c = clean(case_id, "t1c")
    arr_t1c = io_utils.a_numpy(t1c).astype(np.float32)
    arr_t1c_raw = io_utils.a_numpy(raw(case_id, "t1c")).astype(np.float32)
    arr_t1n_raw = io_utils.a_numpy(raw(case_id, "t1n")).astype(np.float32)
    gt = gt_et(case_id, t1c)
    roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=0.5)

    if not roi.any() or _semilla_degenerada(roi):
        pred = (roi > 0).astype(np.uint8)
    else:
        sl = _bbox(roi, margin=12, shape=mapa.shape)
        img = mapa[sl].astype(np.float32)
        img = (img - img.min()) / (np.ptp(img) + 1e-6)
        init = roi[sl].astype(np.uint8)
        _reset_morphsnakes_curvop("first")
        ls = morphological_chan_vese(
            img,
            num_iter=35,
            init_level_set=init,
            smoothing=3,
            lambda1=1.0,
            lambda2=1.0,
        ).astype(np.uint8)
        ls = _orientar_chanvese(ls, img)
        pred = np.zeros_like(mapa, dtype=np.uint8)
        pred[sl] = ls

    return {
        "case_id": case_id,
        "pred": pred,
        "roi": roi.astype(np.uint8),
        "mapa": mapa,
        "cerebro": arr_t1c_raw > 0,
        "gt": gt,
    }


def evaluate_guard(candidate: dict, lcc_threshold: float, enh_threshold: float,
                   vol_threshold: float) -> dict:
    raw_pred = (candidate["pred"] > 0).astype(np.uint8)
    roi = candidate["roi"]
    mapa = candidate["mapa"]
    restr = _restringir_realce(raw_pred, mapa, candidate["cerebro"], 80.0)
    used_restriction = bool(raw_pred.sum() > 0 and restr.sum() >= 0.5 * raw_pred.sum())
    pred = restr if used_restriction else raw_pred
    fallback_reason = ""
    evidence_accept = False
    evidence = {
        "evidence_lcc_fraction": 0.0,
        "evidence_pred_enhancement": 0.0,
        "evidence_init_enhancement": 0.0,
        "evidence_enhancement_ratio": 0.0,
        "evidence_volume_multiple": 0.0,
    }

    if roi.any():
        a_init = float(roi.sum())
        a_pred = float(pred.sum())
        inter = float(np.logical_and(pred > 0, roi > 0).sum())
        if a_pred == 0:
            fallback_reason = "empty_prediction"
        elif a_pred < 0.40 * a_init:
            fallback_reason = "collapsed_small"
        elif a_pred > vol_threshold * a_init:
            fallback_reason = "leaked_large"
        elif inter < 0.40 * a_pred:
            pred_vals = mapa[pred > 0]
            init_vals = mapa[roi > 0]
            pred_enh = float(pred_vals.mean()) if pred_vals.size else 0.0
            init_enh = float(init_vals.mean()) if init_vals.size else 0.0
            enhancement_ratio = pred_enh / max(init_enh, 1e-6)
            volume_multiple = a_pred / max(a_init, 1.0)
            lcc = lcc_fraction(pred)
            evidence = {
                "evidence_lcc_fraction": lcc,
                "evidence_pred_enhancement": pred_enh,
                "evidence_init_enhancement": init_enh,
                "evidence_enhancement_ratio": enhancement_ratio,
                "evidence_volume_multiple": volume_multiple,
            }
            evidence_accept = (
                lcc >= lcc_threshold
                and enhancement_ratio >= enh_threshold
                and volume_multiple <= vol_threshold
            )
            if not evidence_accept:
                fallback_reason = "low_seed_overlap"

        if fallback_reason:
            pred = (roi > 0).astype(np.uint8)

    if pred.any():
        pred = ndimage.binary_fill_holes(pred).astype(np.uint8)
        pred = _morfologia(pred, erosion=0, dilatacion=0, keep_largest=True)

    equals_roi = bool(roi.any() and np.array_equal(pred > 0, roi > 0))
    branch = (
        "collapse-detected" if fallback_reason else
        "ROI-fallback" if equals_roi else
        "evolved"
    )
    reason = (
        fallback_reason if fallback_reason else
        "final_equals_init_no_collapse" if equals_roi else
        "accepted_evidence_seed_divergence" if evidence_accept else
        "accepted"
    )
    return {
        "dice": round(float(dice(pred, candidate["gt"])), 4),
        "branch": branch,
        "reason": reason,
        "voxels": int(pred.sum()),
        "evidence_accept": bool(evidence_accept),
        **evidence,
    }


def summarize_setting(sweep_param: str, sweep_value: float,
                      lcc_threshold: float, enh_threshold: float,
                      vol_threshold: float, candidates: list[dict],
                      baseline_vs: pd.DataFrame, sacred: set[str]) -> dict:
    rows = []
    for candidate in candidates:
        case_id = candidate["case_id"]
        base = baseline_vs.loc[case_id]
        result = evaluate_guard(candidate, lcc_threshold, enh_threshold, vol_threshold)
        delta = result["dice"] - float(base["dice_ET"])
        rows.append({
            "case_id": case_id,
            "baseline_dice": float(base["dice_ET"]),
            "stage3a_dice": result["dice"],
            "delta": delta,
            "baseline_branch": base["guard_branch"],
            "stage3a_branch": result["branch"],
            "stage3a_reason": result["reason"],
            "evidence_accept": result["evidence_accept"],
        })

    per_case = pd.DataFrame(rows)
    sacred_rows = per_case[per_case["case_id"].isin(sacred)]
    sacred_flipped = sacred_rows[sacred_rows["stage3a_branch"] != "evolved"]
    sacred_lost = sacred_rows[sacred_rows["delta"] < -1e-9]
    improved = per_case[per_case["delta"] > 1e-9]
    worsened = per_case[per_case["delta"] < -1e-9]
    changed = per_case[
        (per_case["delta"].abs() > 1e-9)
        | (per_case["baseline_branch"] != per_case["stage3a_branch"])
    ]
    return {
        "sweep_param": sweep_param,
        "sweep_value": sweep_value,
        "lcc_threshold": lcc_threshold,
        "enh_ratio_threshold": enh_threshold,
        "vol_mult_threshold": vol_threshold,
        "blended_dice": float(per_case["stage3a_dice"].mean()),
        "baseline_blended_dice": float(per_case["baseline_dice"].mean()),
        "blended_delta": float(per_case["stage3a_dice"].mean() - per_case["baseline_dice"].mean()),
        "sacred_flip_count": int(len(sacred_flipped)),
        "sacred_loss_count": int(len(sacred_lost)),
        "sacred_flip_cases": ";".join(sacred_flipped["case_id"]),
        "sacred_loss_cases": ";".join(sacred_lost["case_id"]),
        "recovers_02116": bool(
            per_case.loc[per_case["case_id"] == "BraTS-GLI-02116-100", "delta"].iloc[0] > 0
        ),
        "changed_cases": ";".join(changed["case_id"]),
        "improved_cases": ";".join(improved["case_id"]),
        "worsened_cases": ";".join(worsened["case_id"]),
        "evidence_accepted_cases": ";".join(
            per_case.loc[per_case["evidence_accept"], "case_id"]
        ),
    }


def main() -> None:
    baseline = pd.read_csv(BASELINE)
    baseline_vs = (
        baseline[baseline["metodo"] == "variational_spline"]
        .set_index("case_id")
        .sort_index()
    )
    sacred = set(baseline_vs[baseline_vs["guard_branch"] == "evolved"].index)
    candidates = [raw_variational_candidate(case_id) for case_id in baseline_vs.index]

    settings = []
    for value in ENH_GRID:
        settings.append(("enh_ratio", value, FIXED_LCC, value, FIXED_VOL))
    for value in LCC_GRID:
        settings.append(("lcc_frac", value, value, FIXED_ENH, FIXED_VOL))
    for value in VOL_GRID:
        settings.append(("vol_mult", value, FIXED_LCC, FIXED_ENH, value))

    rows = [
        summarize_setting(name, value, lcc, enh, vol, candidates, baseline_vs, sacred)
        for name, value, lcc, enh, vol in settings
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
