import json
import hashlib
import subprocess
import sys
from pathlib import Path

import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "analysis" / "baseline" / "run_case_hashes.py"
CASE_ID = "BraTS-GLI-02108-100"


def _run_once():
    proc = subprocess.run(
        [sys.executable, str(RUNNER), CASE_ID],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_case_masks_are_bit_identical_across_subprocess_runs():
    run1 = _run_once()
    run2 = _run_once()
    assert run1["methods"].keys() == run2["methods"].keys()
    for method in run1["methods"]:
        assert run1["methods"][method]["hash"] == run2["methods"][method]["hash"], method


def _mask_hash(path: Path) -> str:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _run_runner(output_root: Path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_all.py"),
            "--config",
            str(ROOT / "configs" / "pipeline.yaml"),
            "--case-id",
            CASE_ID,
            "--skip-clean",
            "--skip-viz",
            "--skip-poisson",
            "--clean-root",
            str(ROOT / "output" / "limpieza"),
            "--output-root",
            str(output_root),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


def test_runner_masks_are_bit_identical_across_subprocess_runs(tmp_path):
    out1 = tmp_path / "runner1"
    out2 = tmp_path / "runner2"
    _run_runner(out1)
    _run_runner(out2)
    seg1 = out1 / "segmentacion" / CASE_ID
    seg2 = out2 / "segmentacion" / CASE_ID
    files1 = sorted(p.name for p in seg1.glob(f"{CASE_ID}-et_*.nii.gz"))
    files2 = sorted(p.name for p in seg2.glob(f"{CASE_ID}-et_*.nii.gz"))
    assert files1 == files2
    for name in files1:
        assert _mask_hash(seg1 / name) == _mask_hash(seg2 / name), name
