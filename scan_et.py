"""Scan source dataset for ET (label 3) volume per case; print top cases.
Used once to pick the 20-case evaluation subset (one timepoint per patient)."""
import os, glob, sys
import numpy as np
import nibabel as nib

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 120

cases = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
seen_patients = set()
rows = []
for c in cases:
    patient = c.rsplit("-", 1)[0]          # collapse timepoints -> 1 per patient
    if patient in seen_patients:
        continue
    seg = glob.glob(os.path.join(SRC, c, "*-seg.nii*"))
    if not seg:
        continue
    arr = np.asarray(nib.load(seg[0]).dataobj)
    et = int((np.round(arr) == 3).sum())
    tot = int((arr > 0).sum())
    rows.append((c, et, tot))
    seen_patients.add(patient)
    if len(seen_patients) >= LIMIT:
        break

rows.sort(key=lambda r: r[1], reverse=True)
print(f"scanned {len(rows)} patients; top 30 by ET voxel count:")
for c, et, tot in rows[:30]:
    print(f"{c:28s} ET={et:7d}  WT={tot:7d}  ET/WT={et/max(tot,1):.3f}")
