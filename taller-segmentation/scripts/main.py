#!/usr/bin/env python
"""main.py — Unsupervised K-means segmentation pipeline driver.

Runs the full pipeline on the 3 MR volumes (brain, breast, liver):

    VolumeIO -> ForegroundMaskBuilder -> IntensityPreprocessor
        -> for K in 2..8:
               for seed_strategy in {quantile_anchored, warm_start}:
                   KMeansRunner + SilhouetteEvaluator
        -> pick best (K, strategy) by mean silhouette
        -> LesionMaskExtractor -> PostProcessor
        -> write label map + lesion mask + experiment_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import (  # noqa: E402
    ForegroundMaskBuilder,
    IntensityPreprocessor,
    KMeansRunner,
    LesionMaskExtractor,
    PostProcessor,
    SeedPlanner,
    SilhouetteEvaluator,
    VolumeIO,
)

ROOT = HERE.parent
IMAGES_DIR = ROOT / "images"
RESULTS_DIR = ROOT / "results" / "unsupervised_kmeans"
JSON_PATH = ROOT / "experiment_results.json"

VOLUMES = [
    ("brain", IMAGES_DIR / "MRBrainTumor.nii.gz"),
    ("breast", IMAGES_DIR / "MRBreastCancer.nii.gz"),
    ("liver", IMAGES_DIR / "MRLiverTumor.nii.gz"),
]

# K=2 always collapses to tissue-vs-background on MR histograms. Start at K=3
# so the lesion has a chance to separate from healthy parenchyma.
K_SWEEP = list(range(3, 9))

ORGAN_CONFIGS = {
    "brain": {
        "mask_builder": dict(
            threshold_method="otsu", closing_radius=3, opening_radius=1,
            keep_largest=True, fill_holes=True, inner_erode_radius=6,
        ),
        "lesion": dict(
            min_cluster_fraction=0.0005, max_cluster_fraction=0.10,
            prefer_tail="bright",
        ),
    },
    "breast": {
        "mask_builder": dict(
            threshold_method="li", closing_radius=5, opening_radius=2,
            keep_largest=False, min_component_voxels=20000, fill_holes=True,
            min_voxels=200000,
        ),
        "lesion": dict(
            min_cluster_fraction=0.0005, max_cluster_fraction=0.08,
            prefer_tail="bright",
        ),
    },
    "liver": {
        "mask_builder": dict(
            threshold_method="otsu", closing_radius=3, opening_radius=1,
            keep_largest=True, fill_holes=True,
        ),
        "lesion": dict(
            min_cluster_fraction=0.0005, max_cluster_fraction=0.10,
            prefer_tail="bright",
        ),
    },
}


def process_volume(name: str, path: Path) -> dict:
    print(f"\n{'='*72}\n  Processing: {name}  ({path.name})\n{'='*72}")
    t0 = time.perf_counter()

    image = VolumeIO.read(path)
    geometry = VolumeIO.geometry(image)
    print(f"  Volume size={geometry['size']}  spacing={geometry['spacing']}")

    config = ORGAN_CONFIGS.get(name, ORGAN_CONFIGS["brain"])

    print("  [1/6] Building foreground ROI...")
    mask_builder = ForegroundMaskBuilder(**config["mask_builder"])
    roi_mask = mask_builder.build(image)
    roi_arr = VolumeIO.to_numpy(roi_mask).astype(bool)
    print(f"        ROI voxels: {int(roi_arr.sum()):,}")

    print("  [2/6] Intensity preprocessing (shift + N4 + z-score in ROI)...")
    preprocessor = IntensityPreprocessor(
        n4_iterations=(40, 40, 30), n4_shrink=4, normalization="zscore"
    )
    prep_image, prep_stats = preprocessor.run(image, roi_mask)
    prep_arr = VolumeIO.to_numpy(prep_image)
    roi_values = prep_arr[roi_arr]
    print(f"        post-N4 ROI mean={prep_stats['post_roi_mean']:.3f}")

    gradient_magnitude = SilhouetteEvaluator._gradient_magnitude(prep_arr)

    seed_planner = SeedPlanner(low_anchor=2.0, high_anchor=98.0)
    kmeans_runner = KMeansRunner()
    silhouette = SilhouetteEvaluator(sample_size=20000, edge_fraction=0.5, n_repeats=3)

    sweep_log: list[dict] = []
    best_means_by_k: dict[int, list[float]] = {}
    best_result_by_k: dict[int, dict] = {}

    for k in K_SWEEP:
        previous_means = best_means_by_k.get(k - 1)
        plans = seed_planner.plan(roi_values, k, previous_means=previous_means)
        best_for_this_k = None

        for strategy, seeds in plans.items():
            print(f"  [3/6] K={k}  strategy={strategy}  seeds={[round(s,3) for s in seeds]}")
            run = kmeans_runner.run(prep_image, roi_mask, seeds, strategy)
            metrics = silhouette.evaluate(prep_arr, run.label_array, roi_arr)
            entry = {
                "k": k,
                "strategy": strategy,
                "initial_seeds": seeds,
                "final_means": run.final_means,
                "silhouette_mean": metrics["score"],
                "silhouette_variance": metrics["variance"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "calinski_harabasz_variance": metrics["calinski_harabasz_variance"],
                "silhouette_repeats": metrics.get("n_repeats", 0),
            }
            sweep_log.append(entry)
            ch = metrics["calinski_harabasz"]
            if not np.isnan(ch):
                if best_for_this_k is None or ch > best_for_this_k["calinski_harabasz"]:
                    best_for_this_k = entry
                    best_result_by_k[k] = {
                        "entry": entry,
                        "label_array": run.label_array,
                        "label_image": run.label_image,
                    }
            print(
                f"        CH={ch:.1f}  silhouette={metrics['score']:.4f}  var={metrics['variance']:.5f}"
            )

        if best_for_this_k:
            best_means_by_k[k] = best_for_this_k["final_means"]

    valid_ks = [
        k for k, v in best_result_by_k.items()
        if not np.isnan(v["entry"]["calinski_harabasz"])
    ]
    if not valid_ks:
        raise RuntimeError(f"No valid CH evaluated for {name}")
    best_k = max(valid_ks, key=lambda k: best_result_by_k[k]["entry"]["calinski_harabasz"])
    winner = best_result_by_k[best_k]
    print(
        f"  [4/6] Winner: K={best_k}  CH={winner['entry']['calinski_harabasz']:.1f}  "
        f"silhouette={winner['entry']['silhouette_mean']:.4f}"
    )

    print("  [5/6] Lesion plausibility scoring...")
    extractor = LesionMaskExtractor(**config["lesion"])
    lesion = extractor.extract(
        intensity_array=prep_arr,
        label_array=winner["label_array"],
        roi_mask=roi_arr,
        gradient_magnitude=gradient_magnitude,
    )
    print(f"        Chosen cluster ids: {lesion['chosen_ids']}  reason={lesion['reason']}")

    print("  [6/6] Post-processing (island removal + closing + fill)...")
    post = PostProcessor(min_component_voxels=50, max_components=3, closing_iterations=1)
    final_mask = post.run(lesion["binary_mask"])

    import SimpleITK as sitk  # local import keeps top tidy

    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    label_img = VolumeIO.from_numpy(
        winner["label_array"].astype(np.uint8), image, pixel_type=sitk.sitkUInt8
    )
    VolumeIO.write(label_img, out_dir / "label_map.nii.gz")
    mask_img = VolumeIO.from_numpy(
        final_mask.astype(np.uint8), image, pixel_type=sitk.sitkUInt8
    )
    VolumeIO.write(mask_img, out_dir / "lesion_mask.nii.gz")
    VolumeIO.write(roi_mask, out_dir / "roi_mask.nii.gz")

    elapsed = time.perf_counter() - t0

    k_summary = {}
    for k in K_SWEEP:
        rows = [e for e in sweep_log if e["k"] == k and not np.isnan(e["calinski_harabasz"])]
        if not rows:
            continue
        best_row = max(rows, key=lambda r: r["calinski_harabasz"])
        k_summary[str(k)] = {
            "best_strategy": best_row["strategy"],
            "best_calinski_harabasz": best_row["calinski_harabasz"],
            "best_silhouette_mean": best_row["silhouette_mean"],
            "best_silhouette_variance": best_row["silhouette_variance"],
            "final_means": best_row["final_means"],
        }

    report = {
        "image_name": name,
        "image_path": str(path),
        "geometry": geometry,
        "roi_voxels": int(roi_arr.sum()),
        "preprocessing": prep_stats,
        "evaluated_k": K_SWEEP,
        "sweep": sweep_log,
        "k_summary": k_summary,
        "chosen_k": int(best_k),
        "chosen_strategy": winner["entry"]["strategy"],
        "initial_seeds": winner["entry"]["initial_seeds"],
        "final_means": winner["entry"]["final_means"],
        "winning_silhouette_mean": winner["entry"]["silhouette_mean"],
        "winning_silhouette_variance": winner["entry"]["silhouette_variance"],
        "winning_calinski_harabasz": winner["entry"]["calinski_harabasz"],
        "selection_criterion": "calinski_harabasz",
        "lesion_cluster_ids": lesion["chosen_ids"],
        "lesion_reason": lesion["reason"],
        "lesion_tail": lesion.get("tail"),
        "lesion_cluster_scores": lesion["scores"],
        "lesion_mask_voxels": int(final_mask.sum()),
        "outputs": {
            "label_map": str(out_dir / "label_map.nii.gz"),
            "lesion_mask": str(out_dir / "lesion_mask.nii.gz"),
            "roi_mask": str(out_dir / "roi_mask.nii.gz"),
        },
        "processing_time_seconds": float(elapsed),
    }
    print(f"  Done in {elapsed:.1f}s   lesion voxels={int(final_mask.sum()):,}")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", choices=["brain", "breast", "liver"], default=None,
        help="Process only this volume and merge its result into the existing JSON.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = [(n, p) for n, p in VOLUMES if args.only is None or n == args.only]

    if args.only and JSON_PATH.exists():
        with JSON_PATH.open("r", encoding="utf-8") as f:
            all_reports = json.load(f)
    else:
        all_reports = []

    for name, path in selected:
        try:
            report = process_volume(name, path)
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            report = {"image_name": name, "image_path": str(path), "error": str(exc)}

        replaced = False
        for i, existing in enumerate(all_reports):
            if existing.get("image_name") == name:
                all_reports[i] = report
                replaced = True
                break
        if not replaced:
            all_reports.append(report)

        with JSON_PATH.open("w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\n{'='*72}\n  Processed: {[n for n,_ in selected]}  QC report: {JSON_PATH}\n{'='*72}")


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    main()
