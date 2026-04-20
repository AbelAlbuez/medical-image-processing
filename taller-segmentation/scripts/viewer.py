#!/usr/bin/env python
"""Interactive NIfTI viewer to inspect K-means segmentation outputs.

Usage:
    python scripts/viewer.py brain
    python scripts/viewer.py breast
    python scripts/viewer.py liver

Controls:
    slider or arrow keys  -> change slice
    a / c / s             -> switch axis (axial / coronal / sagittal)
    j                     -> jump to the slice with the most lesion voxels
    r                     -> jump to the slice with the most ROI voxels
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Disable matplotlib's default key bindings so our axis/jump keys aren't stolen
# (e.g. 's' -> save, 'a' -> all_axes, 'r' -> home, 'c'/left/right -> nav).
for _k in (
    "keymap.save", "keymap.all_axes", "keymap.back", "keymap.forward",
    "keymap.home", "keymap.xscale", "keymap.yscale",
    "keymap.pan", "keymap.zoom", "keymap.grid", "keymap.grid_minor",
):
    if _k in plt.rcParams:
        plt.rcParams[_k] = []

import numpy as np
from matplotlib.widgets import Slider

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import VolumeIO  # noqa: E402

ROOT = HERE.parent
IMAGES_DIR = ROOT / "images"
RESULTS_DIR = ROOT / "results" / "unsupervised_kmeans"

VOLUMES = {
    "brain":  IMAGES_DIR / "MRBrainTumor.nii.gz",
    "breast": IMAGES_DIR / "MRBreastCancer.nii.gz",
    "liver":  IMAGES_DIR / "MRLiverTumor.nii.gz",
}

AXIS_LABEL = {0: "axial (z)", 1: "coronal (y)", 2: "sagittal (x)"}


def load_array(path: Path) -> np.ndarray:
    return VolumeIO.to_numpy(VolumeIO.read(path))


def axis_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return vol[idx, :, :]
    if axis == 1:
        return vol[:, idx, :]
    return vol[:, :, idx]


def peak_slice(mask: np.ndarray, axis: int) -> int | None:
    other = tuple(a for a in (0, 1, 2) if a != axis)
    sums = mask.sum(axis=other)
    if sums.sum() == 0:
        return None
    return int(np.argmax(sums))


def report_presence(name: str, mask: np.ndarray, label: str) -> None:
    total = int(mask.sum())
    print(f"  {label}: {total:,} voxels total")
    if total == 0:
        return
    for ax in (0, 1, 2):
        other = tuple(a for a in (0, 1, 2) if a != ax)
        sums = mask.sum(axis=other)
        nz = np.flatnonzero(sums)
        if nz.size:
            print(
                f"    {AXIS_LABEL[ax]:<14} slices {nz.min()}..{nz.max()}  "
                f"peak @ {int(np.argmax(sums))} ({int(sums.max())} voxels)"
            )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in VOLUMES:
        print("usage: viewer.py {brain|breast|liver}")
        sys.exit(1)
    name = sys.argv[1]

    original = load_array(VOLUMES[name])
    out_dir = RESULTS_DIR / name
    label_map = load_array(out_dir / "label_map.nii.gz")
    lesion = load_array(out_dir / "lesion_mask.nii.gz").astype(bool)
    roi = load_array(out_dir / "roi_mask.nii.gz").astype(bool)

    print(f"\n== {name} ==  shape={original.shape}")
    report_presence(name, lesion, "lesion mask")
    report_presence(name, roi, "ROI mask")

    state = {"axis": 0, "idx": original.shape[0] // 2}

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.18, top=0.90)

    def redraw() -> None:
        axis, idx = state["axis"], state["idx"]
        idx = max(0, min(idx, original.shape[axis] - 1))
        state["idx"] = idx

        orig_slice = axis_slice(original, axis, idx)
        lesion_slice = axis_slice(lesion, axis, idx)
        label_slice = axis_slice(label_map, axis, idx)

        for a in axes:
            a.clear()
            a.axis("off")

        axes[0].imshow(orig_slice, cmap="gray")
        axes[0].set_title(f"Original\n{AXIS_LABEL[axis]} idx={idx}", fontsize=10)

        axes[1].imshow(orig_slice, cmap="gray")
        if lesion_slice.any():
            axes[1].imshow(
                np.ma.masked_where(~lesion_slice, lesion_slice),
                cmap="autumn", alpha=0.55,
            )
        axes[1].set_title(
            f"Original + lesion overlay\nlesion voxels in slice = {int(lesion_slice.sum())}",
            fontsize=10,
        )

        axes[2].imshow(label_slice, cmap="tab10")
        axes[2].set_title("K-means label map", fontsize=10)
        fig.canvas.draw_idle()

    max_dim = max(original.shape) - 1
    ax_slider = plt.axes([0.2, 0.06, 0.6, 0.03])
    slider = Slider(ax_slider, "slice", 0, max_dim, valinit=state["idx"], valstep=1)

    def on_slider(val: float) -> None:
        state["idx"] = int(val)
        redraw()

    slider.on_changed(on_slider)

    def set_idx(new_idx: int) -> None:
        new_idx = max(0, min(new_idx, original.shape[state["axis"]] - 1))
        state["idx"] = new_idx
        slider.eventson = False
        slider.set_val(new_idx)
        slider.eventson = True
        redraw()

    def on_key(event) -> None:
        if event.key in ("a", "c", "s"):
            state["axis"] = {"a": 0, "c": 1, "s": 2}[event.key]
            set_idx(original.shape[state["axis"]] // 2)
        elif event.key == "right":
            set_idx(state["idx"] + 1)
        elif event.key == "left":
            set_idx(state["idx"] - 1)
        elif event.key == "j":
            peak = peak_slice(lesion, state["axis"])
            if peak is not None:
                set_idx(peak)
                print(f"  -> jumped to lesion peak: {AXIS_LABEL[state['axis']]} = {peak}")
            else:
                print("  (no lesion voxels to jump to)")
        elif event.key == "r":
            peak = peak_slice(roi, state["axis"])
            if peak is not None:
                set_idx(peak)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.suptitle(
        f"{name}   keys: a/c/s=axis   ←/→=slice   j=jump to lesion peak   r=ROI peak",
        fontsize=11,
    )
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
