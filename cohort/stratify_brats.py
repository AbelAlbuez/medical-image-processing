#!/usr/bin/env python3
"""
stratify_brats.py — characterize the BraTS-2024 GLI training pool and draw a
stratified sample, so the selected cohort is a MEASURED selection, not a guessed list.

Two subcommands:
  characterize  — walk the unzipped training data, compute per-case stratification
                  variables from the seg mask (and optionally the images), write a
                  per-case CSV. Expensive; run once over the full pool.
  sample        — read that CSV, bin cases into strata using pool-wide percentiles,
                  draw a stratified sample (seeded), and emit the markdown manifest.

ET = label 3, RC = label 4 (BraTS-2024 GLI convention; confirmed on this project's disk).

This script has NO dependency on the segmentation pipeline. It only reads NIfTI.
Requires: SimpleITK, numpy, scipy (all already in this project's env).
"""
import argparse, csv, json, os, sys, glob, random, itertools
from collections import defaultdict, Counter

import numpy as np
from scipy import ndimage

ET_LABEL = 3
RC_LABEL = 4
MIN_CC_VOXELS = 10          # ignore ET blobs smaller than this when counting foci
BRAIN_MIN_INTENSITY = 1e-6  # BraTS is skull-stripped: background is 0


# ----------------------------- characterize ------------------------------------

def _load(path):
    import SimpleITK as sitk  # lazy: module imports fine without it; only reading needs it
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)          # (z, y, x)
    spacing_xyz = img.GetSpacing()             # (x, y, z)
    spacing_zyx = spacing_xyz[::-1]            # (z, y, x) — for EDT / anisotropy
    voxel_mm3 = float(np.prod(spacing_xyz))
    return arr, spacing_zyx, voxel_mm3


def _percentile_norm(vol, brain, lo=1.0, hi=99.0):
    """Percentile-clip + rescale to [0,1] inside the brain mask (matches the pipeline)."""
    v = vol.astype(np.float32)
    inside = v[brain]
    if inside.size == 0:
        return np.zeros_like(v)
    p_lo, p_hi = np.percentile(inside, [lo, hi])
    if p_hi <= p_lo:
        return np.zeros_like(v)
    out = np.clip((v - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    out[~brain] = 0.0
    return out


def characterize_case(case_id, seg_path, paths, with_images):
    """Return a dict of measured stratification variables for one case."""
    row = {"case_id": case_id, "error": ""}
    try:
        seg, spacing_zyx, voxel_mm3 = _load(seg_path)
    except Exception as e:
        row["error"] = f"seg_load_failed: {e}"
        return row

    labels_present = sorted(int(x) for x in np.unique(seg))
    row["labels_present"] = " ".join(str(x) for x in labels_present)
    row["shape"] = "x".join(str(s) for s in seg.shape)
    row["voxel_mm3"] = round(voxel_mm3, 6)

    et = (seg == ET_LABEL)
    rc = (seg == RC_LABEL)
    et_vox = int(et.sum())
    rc_vox = int(rc.sum())
    row["et_voxels"] = et_vox
    row["et_mm3"] = round(et_vox * voxel_mm3, 3)
    row["et_present"] = int(et_vox > 0)
    row["rc_voxels"] = rc_vox
    row["rc_present"] = int(rc_vox > 0)

    # --- Focality: connected components of ET (26-connectivity), noise-filtered ---
    if et_vox > 0:
        structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
        lbl, n = ndimage.label(et, structure=structure)
        if n > 0:
            sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
            kept = int((sizes >= MIN_CC_VOXELS).sum())
            row["et_num_components"] = max(kept, 1)
            row["et_largest_frac"] = round(float(sizes.max()) / et_vox, 4)
        else:
            row["et_num_components"] = 1
            row["et_largest_frac"] = 1.0
    else:
        row["et_num_components"] = 0
        row["et_largest_frac"] = 0.0
    row["multifocal"] = int(row["et_num_components"] >= 2)

    # --- Peri-cavity confound proxy (seg-only): min ET->RC distance in mm ---
    # This is the 00533/02078 axis without loading images: ET hugging the cavity margin.
    if et_vox > 0 and rc_vox > 0:
        # EDT of the complement of RC gives, at each voxel, distance to nearest RC voxel.
        dist_to_rc = ndimage.distance_transform_edt(~rc, sampling=spacing_zyx)
        row["et_to_rc_min_mm"] = round(float(dist_to_rc[et].min()), 3)
    else:
        row["et_to_rc_min_mm"] = ""  # undefined if either is absent

    # --- Enhancement-confound proxy (needs t1c + t1n): is the brightest ---
    # --- enhancement actually ET, or a disjoint non-tumor structure? -------------
    # Mirrors the Stage 3D diagnostic that found the irreducible failures.
    if with_images and et_vox > 0 and paths.get("t1c") and paths.get("t1n"):
        try:
            t1c, _, _ = _load(paths["t1c"])
            t1n, _, _ = _load(paths["t1n"])
            brain = t1c > BRAIN_MIN_INTENSITY
            mapa = _percentile_norm(t1c, brain) - _percentile_norm(t1n, brain)
            mapa[~brain] = 0.0
            pos = mapa[brain]
            pos = pos[pos > 0]
            if pos.size > 50:
                thr = np.percentile(pos, 90.0)
                cand = (mapa >= thr) & brain
                if cand.sum() > 0:
                    lblc, nc = ndimage.label(cand, structure=np.ones((3, 3, 3), int))
                    sizes = ndimage.sum(np.ones_like(lblc), lblc, index=range(1, nc + 1))
                    top = int(np.argmax(sizes)) + 1
                    top_blob = (lblc == top)
                    overlap = float((top_blob & et).sum()) / float(top_blob.sum())
                    row["enh_top_overlap_et"] = round(overlap, 4)
                    # high confound risk = brightest enhancement is NOT ET
                    row["enh_confound_risk"] = round(1.0 - overlap, 4)
                    row["enh_mean_in_et"] = round(float(mapa[et].mean()), 4)
                    row["enh_mean_in_top"] = round(float(mapa[top_blob].mean()), 4)
        except Exception as e:
            row["error"] = f"image_confound_failed: {e}"

    return row


def find_cases(root):
    """Find all cases by their -seg.nii.gz files; derive case id + sibling modalities."""
    seg_files = sorted(glob.glob(os.path.join(root, "**", "*-seg.nii.gz"), recursive=True))
    cases = []
    for seg in seg_files:
        base = os.path.basename(seg)[:-len("-seg.nii.gz")]  # strip suffix -> case id
        d = os.path.dirname(seg)
        paths = {}
        for mod in ("t1c", "t1n", "t2w", "t2f"):
            p = os.path.join(d, f"{base}-{mod}.nii.gz")
            if os.path.exists(p):
                paths[mod] = p
        cases.append((base, seg, paths))
    return cases


def cmd_characterize(args):
    cases = find_cases(args.data_root)
    if not cases:
        print(f"ERROR: no *-seg.nii.gz found under {args.data_root}", file=sys.stderr)
        sys.exit(2)
    print(f"Found {len(cases)} cases under {args.data_root}")
    if args.limit:
        cases = cases[:args.limit]
        print(f"Limiting to first {len(cases)} (--limit)")

    rows = []
    for i, (cid, seg, paths) in enumerate(cases, 1):
        row = characterize_case(cid, seg, paths, with_images=args.with_images)
        rows.append(row)
        if i % 25 == 0 or i == len(cases):
            print(f"  [{i}/{len(cases)}] {cid} "
                  f"et={row.get('et_voxels','?')} foci={row.get('et_num_components','?')} "
                  f"rc={row.get('rc_present','?')} "
                  f"{'ERR '+row['error'] if row.get('error') else ''}")

    # union of keys across rows -> stable column order
    preferred = ["case_id", "et_present", "et_voxels", "et_mm3", "et_num_components",
                 "multifocal", "et_largest_frac", "rc_present", "rc_voxels",
                 "et_to_rc_min_mm", "enh_top_overlap_et", "enh_confound_risk",
                 "enh_mean_in_et", "enh_mean_in_top", "labels_present", "shape",
                 "voxel_mm3", "error"]
    keys = preferred + [k for r in rows for k in r if k not in preferred]
    seen, cols = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); cols.append(k)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"\nWrote {args.out}: {len(rows)} cases ({n_err} with errors).")
    print("Next: python stratify_brats.py sample --pool", args.out)


# -------------------------------- sample ---------------------------------------

def _vol_bin(et_present, et_mm3, p33, p66):
    if not et_present:
        return "absent"
    if et_mm3 < p33:
        return "small"
    if et_mm3 < p66:
        return "medium"
    return "large"


def _confound_bin(row, near_mm):
    """Seg-only peri-cavity confound bin. 'near' if ET hugs the resection margin."""
    if str(row.get("rc_present", "0")) != "1":
        return "no_rc"
    d = row.get("et_to_rc_min_mm", "")
    try:
        d = float(d)
    except (TypeError, ValueError):
        return "no_rc"
    return "near_rc" if d <= near_mm else "far_rc"


def cmd_sample(args):
    with open(args.pool) as f:
        rows = [r for r in csv.DictReader(f) if not r.get("error")]
    # numeric coercions
    for r in rows:
        r["et_present"] = int(r.get("et_present", "0") or 0)
        r["et_mm3"] = float(r.get("et_mm3", "0") or 0.0)
        r["multifocal"] = int(r.get("multifocal", "0") or 0)

    et_present_mm3 = sorted(r["et_mm3"] for r in rows if r["et_present"])
    if len(et_present_mm3) < 3:
        print("ERROR: too few ET-present cases to bin.", file=sys.stderr); sys.exit(2)
    p33, p66 = np.percentile(et_present_mm3, [33.3, 66.6])

    # assign strata
    for r in rows:
        r["vol_bin"] = _vol_bin(r["et_present"], r["et_mm3"], p33, p66)
        r["confound_bin"] = _confound_bin(r, args.near_mm)
        r["focality"] = "multifocal" if r["multifocal"] else "unifocal"
        r["stratum"] = f"{r['vol_bin']}|{r['focality']}|{r['confound_bin']}"

    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)

    print(f"Pool: {len(rows)} usable cases. ET-volume p33={p33:.0f} p66={p66:.0f} mm3")
    print("Stratum populations (available):")
    for s in sorted(by_stratum, key=lambda k: -len(by_stratum[k])):
        print(f"  {len(by_stratum[s]):4d}  {s}")

    # ---- allocation: guarantee minimums on the hard/interesting strata, then fill
    rng = random.Random(args.seed)
    target_pool = args.pool_size          # 150
    guaranteed = {  # substrings that must be over-represented (failure-relevant)
        "absent": args.min_absent,
        "small": args.min_small,
        "multifocal": args.min_multifocal,
        "near_rc": args.min_near_rc,
    }
    selected, used = [], set()

    def draw_matching(substr, k):
        pool = [r for s, rs in by_stratum.items() if substr in s
                for r in rs if r["case_id"] not in used]
        rng.shuffle(pool)
        take = pool[:k]
        for r in take:
            used.add(r["case_id"]); selected.append(r)
        return len(take)

    for substr, k in guaranteed.items():
        got = draw_matching(substr, k)
        print(f"guarantee {substr}: requested {k}, got {got}")

    # fill remainder proportionally to available stratum sizes
    remaining_slots = max(0, target_pool - len(selected))
    fill_pool = [r for rs in by_stratum.values() for r in rs if r["case_id"] not in used]
    rng.shuffle(fill_pool)
    selected.extend(fill_pool[:remaining_slots])
    for r in fill_pool[:remaining_slots]:
        used.add(r["case_id"])

    # ---- mark PROCESS subset (100) preserving stratum proportions, + stratified folds
    rng2 = random.Random(args.seed + 1)
    by_s_sel = defaultdict(list)
    for r in selected:
        by_s_sel[r["stratum"]].append(r)
    process = []
    frac = min(1.0, args.process_size / max(1, len(selected)))
    for s, rs in by_s_sel.items():
        rng2.shuffle(rs)
        n_take = max(1, round(len(rs) * frac)) if rs else 0
        for r in rs[:n_take]:
            r["process"] = 1
        for r in rs[n_take:]:
            r["process"] = 0
        process.extend(rs[:n_take])
    # trim/pad process to exactly process_size
    proc_sorted = [r for r in selected if r.get("process")]
    # assign balanced k-folds to process cases (for train/test of Phase 2 priors).
    # We stratify jointly on ET-present/absent x volume bin, then choose rotated
    # round-robin starts so remainder cases do not all accumulate in fold 0.
    process_cases = [r for r in selected if r.get("process")]
    fold_groups = defaultdict(list)
    for r in process_cases:
        fold_groups[(int(r.get("et_present", 0)), r.get("vol_bin", ""))].append(r)
    ordered_groups = sorted(fold_groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    for i, (_, rs) in enumerate(ordered_groups):
        random.Random(args.seed + 2000 + i).shuffle(rs)

    best = None
    for starts in itertools.product(range(args.folds), repeat=len(ordered_groups)):
        fold_by_case = {}
        for (_, rs), start in zip(ordered_groups, starts):
            for i, r in enumerate(rs):
                fold_by_case[r["case_id"]] = (start + i) % args.folds

        totals = Counter(fold_by_case.values())
        et_present_counts = Counter(
            fold_by_case[r["case_id"]] for r in process_cases if int(r.get("et_present", 0)) == 1)
        et_absent_counts = Counter(
            fold_by_case[r["case_id"]] for r in process_cases if int(r.get("et_present", 0)) == 0)
        vol_spread = 0
        for vol_name in ("absent", "small", "medium", "large"):
            vals = [
                sum(1 for r in process_cases
                    if r.get("vol_bin") == vol_name and fold_by_case[r["case_id"]] == f)
                for f in range(args.folds)
            ]
            vol_spread += max(vals) - min(vals)
        score = (
            max(totals[f] for f in range(args.folds)) - min(totals[f] for f in range(args.folds)),
            max(et_present_counts[f] for f in range(args.folds)) - min(et_present_counts[f] for f in range(args.folds)),
            max(et_absent_counts[f] for f in range(args.folds)) - min(et_absent_counts[f] for f in range(args.folds)),
            vol_spread,
            starts,
        )
        if best is None or score < best[0]:
            best = (score, fold_by_case)

    _, fold_by_case = best
    for r in process_cases:
        r["fold"] = fold_by_case[r["case_id"]]
    for r in selected:
        r.setdefault("process", 0)
        r.setdefault("fold", "")

    _write_manifest(args, selected, p33, p66)
    _write_selected_csv(args, selected)


def _write_selected_csv(args, selected):
    cols = ["case_id", "process", "fold", "stratum", "vol_bin", "focality",
            "confound_bin", "et_present", "et_mm3", "et_num_components",
            "rc_present", "et_to_rc_min_mm", "enh_confound_risk"]
    out = os.path.splitext(args.manifest)[0] + "_selected.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(selected, key=lambda x: x["case_id"]):
            w.writerow({k: r.get(k, "") for k in cols})
    print("Wrote", out)


def _write_manifest(args, selected, p33, p66):
    n_proc = sum(1 for r in selected if r.get("process"))
    strata = Counter(r["stratum"] for r in selected)
    vol = Counter(r["vol_bin"] for r in selected)
    foc = Counter(r["focality"] for r in selected)
    conf = Counter(r["confound_bin"] for r in selected)

    lines = []
    lines.append(f"# BraTS-2024 GLI — Stratified Cohort Manifest\n")
    lines.append("Generated by `stratify_brats.py sample`. This file is the OUTPUT of a "
                 "measurement pass over the full training pool — every stratification "
                 "variable was computed from the segmentation masks (and images, if "
                 "`--with-images` was used), not assigned by hand.\n")
    lines.append("## How to use / extend this manifest\n")
    lines.append("- **Find a case in the data:** each `case_id` is a folder in the "
                 "unzipped `BraTS2024-BraTS-GLI-TrainingData`, e.g. "
                 "`<case_id>/<case_id>-t1c.nii.gz`.\n")
    lines.append("- **Source:** BraTS-2024 GLI (post-treatment) training data, gated via "
                 "Synapse (registration + data-use agreement required). Case IDs are "
                 "specific to that release.\n")
    lines.append("- **Add more cases:** run `characterize` on the full pool, then re-run "
                 "`sample` with a larger `--pool-size` / different `--seed`, or manually "
                 "append rows below with their measured variables (keep the columns).\n")
    lines.append("- **Reproducibility:** seed = "
                 f"`{args.seed}`, pool-size = `{args.pool_size}`, process-size = "
                 f"`{args.process_size}`, near_rc threshold = `{args.near_mm}` mm, "
                 f"ET-vol bins p33/p66 = `{p33:.0f}`/`{p66:.0f}` mm³.\n")
    lines.append("## Stratification axes\n")
    lines.append("| axis | definition | bins |\n|---|---|---|\n"
                 "| ET volume | label-3 volume in mm³ | absent / small (<p33) / "
                 "medium / large (≥p66) |\n"
                 "| Focality | # ET connected components (26-conn, ≥10 vox) | unifocal / "
                 "multifocal |\n"
                 "| Peri-cavity confound | min ET→RC (label-4) distance | near_rc "
                 f"(≤{args.near_mm}mm) / far_rc / no_rc |\n"
                 "| Enhancement confound* | 1 − overlap(brightest enh. blob, ET) | "
                 "recorded per case if `--with-images` |\n")
    lines.append("_*The enhancement-confound column is the direct analogue of the "
                 "Stage 3D mechanism that produced the two irreducible failures "
                 "(brightest enhancement is non-tumor)._\n")
    lines.append("## Summary of selected cohort\n")
    lines.append(f"- Total selected (pool): **{len(selected)}**  |  "
                 f"marked for processing: **{n_proc}**  |  "
                 f"stratified folds: **{args.folds}** (on process set)\n")
    lines.append(f"- Volume: {dict(vol)}\n")
    lines.append(f"- Focality: {dict(foc)}\n")
    lines.append(f"- Confound: {dict(conf)}\n")
    lines.append("\n### Stratum counts\n")
    lines.append("| stratum (vol \\| focality \\| confound) | n |\n|---|---|\n")
    for s in sorted(strata, key=lambda k: -strata[k]):
        lines.append(f"| {s} | {strata[s]} |\n")

    lines.append("\n## Cases\n")
    lines.append("`process=1` marks the working set; `fold` is the stratified CV fold "
                 "(build Phase-2 priors on training folds, evaluate on held-out fold — "
                 "never build a prior on a case you then score).\n\n")
    lines.append("| case_id | process | fold | vol | focality | confound | ET mm³ | "
                 "foci | RC | ET→RC mm |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(selected, key=lambda x: (-int(x.get("process", 0)), x["case_id"])):
        lines.append(
            f"| {r['case_id']} | {r.get('process','')} | {r.get('fold','')} | "
            f"{r['vol_bin']} | {r['focality']} | {r['confound_bin']} | "
            f"{r.get('et_mm3','')} | {r.get('et_num_components','')} | "
            f"{r.get('rc_present','')} | {r.get('et_to_rc_min_mm','')} |\n")

    with open(args.manifest, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Wrote", args.manifest)


# --------------------------------- cli -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("characterize", help="measure every case in the pool")
    c.add_argument("--data-root", required=True,
                   help="path to unzipped BraTS2024-BraTS-GLI-TrainingData")
    c.add_argument("--out", default="pool_characterization.csv")
    c.add_argument("--with-images", action="store_true",
                   help="also compute the enhancement-confound proxy (loads t1c+t1n; ~3x slower)")
    c.add_argument("--limit", type=int, default=0, help="debug: only first N cases")
    c.set_defaults(func=cmd_characterize)

    s = sub.add_parser("sample", help="draw a stratified sample + write manifest")
    s.add_argument("--pool", required=True, help="pool_characterization.csv from characterize")
    s.add_argument("--manifest", default="COHORT_MANIFEST.md")
    s.add_argument("--pool-size", type=int, default=150)
    s.add_argument("--process-size", type=int, default=100)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--near-mm", type=float, default=5.0,
                   help="ET-to-RC distance (mm) below which a case counts as near_rc")
    s.add_argument("--seed", type=int, default=1337)
    s.add_argument("--min-absent", type=int, default=15)
    s.add_argument("--min-small", type=int, default=25)
    s.add_argument("--min-multifocal", type=int, default=25)
    s.add_argument("--min-near-rc", type=int, default=20)
    s.set_defaults(func=cmd_sample)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
