from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "fallback_02116_determinism.csv"
CASE_ID = "BraTS-GLI-02116-100"


CODE = r"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path.cwd()
os.environ.setdefault("BRATS_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("BRATS_DATASET_DIR", str(ROOT / "images"))
os.environ["BRATS_ENABLE_EVIDENCE_GUARD"] = "0"
os.environ["BRATS_ENABLE_BEST_ITERATE"] = "0"
sys.path.insert(0, str(ROOT / "src"))

from brats_pipeline.seg_et_pipeline import correr_pipeline_et

case_id = "BraTS-GLI-02116-100"

def read(path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(path), pixel_type)

def clean(mod):
    return read(ROOT / "output" / "limpieza" / case_id / f"{case_id}-{mod}.nii.gz")

def raw(mod):
    return read(ROOT / "images" / case_id / f"{case_id}-{mod}.nii.gz")

masks, _, df = correr_pipeline_et(
    clean("t1c"),
    clean("t1n"),
    sitk.ReadImage(str(ROOT / "images" / case_id / f"{case_id}-seg.nii.gz")),
    t1c_raw=raw("t1c"),
    t1n_raw=raw("t1n"),
    t2f=clean("t2f"),
    semilla_zyx=None,
    case_id=case_id,
    auto_pct=90.0,
    sigma=0.5,
    verbose=False,
)
arr = np.ascontiguousarray((masks["variational_spline"] > 0).astype("uint8"))
row = df[df["metodo"] == "variational_spline"].iloc[0]
print(json.dumps({
    "case_id": case_id,
    "method": "variational_spline",
    "hash": hashlib.sha256(arr.tobytes()).hexdigest(),
    "voxels": int(arr.sum()),
    "dice_ET": float(row["dice_ET"]),
    "guard_branch": row["guard_branch"],
    "guard_reason": row["guard_reason"],
}))
"""


def main() -> None:
    rows = []
    for run in range(1, 4):
        env = os.environ.copy()
        env["BRATS_ENABLE_EVIDENCE_GUARD"] = "0"
        env["BRATS_ENABLE_BEST_ITERATE"] = "0"
        proc = subprocess.run(
            [sys.executable, "-c", CODE],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        payload["run"] = run
        rows.append(payload)
    out = pd.DataFrame(rows)
    out["bit_identical_to_run1"] = out["hash"] == out.loc[0, "hash"]
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    if not out["bit_identical_to_run1"].all():
        raise SystemExit("fallback path is nondeterministic")


if __name__ == "__main__":
    main()
