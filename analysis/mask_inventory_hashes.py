from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = ROOT / "output" / "segmentacion"


def mask_hash(path: Path) -> tuple[str, int]:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
    return hashlib.sha256(arr.astype("uint8").tobytes()).hexdigest(), int(arr.sum())


def snapshot(out: Path) -> None:
    rows = []
    for path in sorted(SEG_ROOT.glob("*/*-et_*.nii.gz")):
        case_id = path.parent.name
        method = path.name.split("-et_", 1)[1].replace(".nii.gz", "")
        h, vox = mask_hash(path)
        rows.append(
            {
                "case_id": case_id,
                "method": method,
                "path": str(path.relative_to(ROOT)),
                "hash": h,
                "voxels": vox,
                "mtime": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


def diff(old: Path, new: Path, out: Path) -> None:
    a = pd.read_csv(old)
    b = pd.read_csv(new)
    merged = a.merge(b, on=["case_id", "method"], suffixes=("_old", "_new"))
    changed = merged[merged["hash_old"] != merged["hash_new"]].copy()
    changed["voxel_delta_new_minus_old"] = changed["voxels_new"] - changed["voxels_old"]
    cols = [
        "case_id",
        "method",
        "voxels_old",
        "voxels_new",
        "voxel_delta_new_minus_old",
        "hash_old",
        "hash_new",
        "mtime_old",
        "mtime_new",
    ]
    changed[cols].sort_values(["case_id", "method"]).to_csv(out, index=False)
    print(f"wrote {out}")
    print(changed[cols].sort_values(["case_id", "method"]).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("out", type=Path)
    dif = sub.add_parser("diff")
    dif.add_argument("old", type=Path)
    dif.add_argument("new", type=Path)
    dif.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.cmd == "snapshot":
        snapshot(args.out)
    else:
        diff(args.old, args.new, args.out)


if __name__ == "__main__":
    main()
