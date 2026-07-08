"""Run fixture cases 5x in-process and 5x subprocess; compare mask hashes."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_case_hashes import ROOT, run_case


CASES = ["BraTS-GLI-02108-100", "BraTS-GLI-00020-100"]
OUT = ROOT / "analysis" / "baseline" / "determinism_5x_same_5x_subprocess.csv"


def run_subprocess(case_id: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "baseline" / "run_case_hashes.py"), case_id],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    rows = []
    for case_id in CASES:
        results = []
        for i in range(5):
            results.append(("same_process", i + 1, run_case(case_id)))
        for i in range(5):
            results.append(("subprocess", i + 1, run_subprocess(case_id)))

        reference = results[0][2]["methods"]
        for mode, run_idx, result in results:
            for method, payload in sorted(result["methods"].items()):
                ref_hash = reference[method]["hash"]
                rows.append(
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "run": run_idx,
                        "method": method,
                        "hash": payload["hash"],
                        "reference_hash": ref_hash,
                        "bit_identical_to_reference": payload["hash"] == ref_hash,
                        "dice": payload["dice"],
                        "voxels": payload["voxels"],
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"all_bit_identical={bool(out['bit_identical_to_reference'].all())}")
    print(f"wrote={OUT}")


if __name__ == "__main__":
    main()
