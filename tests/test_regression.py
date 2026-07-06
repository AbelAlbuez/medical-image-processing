import json
import os
import sys
from pathlib import Path

import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline.seg_et_pipeline import correr_pipeline_et  # noqa: E402


FIXTURE = ROOT / "analysis" / "baseline" / "regression_fixture.json"


def _read(path: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)


def _clean(case_id: str, mod: str):
    return _read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")


def _raw(case_id: str, mod: str):
    return _read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")


def _gt(case_id: str):
    return sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz"))


def _run_case(case_id: str):
    _, _, df = correr_pipeline_et(
        _clean(case_id, "t1c"),
        _clean(case_id, "t1n"),
        _gt(case_id),
        t1c_raw=_raw(case_id, "t1c"),
        t1n_raw=_raw(case_id, "t1n"),
        t2f=_clean(case_id, "t2f"),
        semilla_zyx=None,
        case_id=case_id,
        auto_pct=90.0,
        sigma=0.5,
        verbose=False,
    )
    return {row.metodo: float(row.dice_ET) for row in df.itertuples(index=False)}


def test_regression_fixture_dice_matches_baseline():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tol = float(fixture["tolerance"])
    for case_id, spec in fixture["cases"].items():
        observed = _run_case(case_id)
        expected = spec["dice_ET"]
        assert set(observed) == set(expected)
        for method, expected_dice in expected.items():
            assert abs(observed[method] - expected_dice) <= tol, (
                case_id,
                method,
                observed[method],
                expected_dice,
            )
