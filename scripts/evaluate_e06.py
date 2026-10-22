"""Evaluate E06 separately on 22 held-out examples from each of six geologies.

Uses exact E05 evaluation metrics, forward-consistency and density-section
plotting routines from scripts/predict_unet.py; doesn't change that file.

Run from repository root:
  python scripts/evaluate_e06.py --visualize-count 3
  python scripts/evaluate_e06.py --cases E05A E05F --visualize-count 22
  python scripts/evaluate_e06.py --visualize-count 0

Outputs: outputs/ml/predict/E06/E06A...E06F/predictions/test_summary.json
Also outputs overall CSV + per-category aggregate summary at predict/E06/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (ROOT, SRC):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from gravity_migration_ml.ml.dataset import load_sample
from gravity_migration_ml.ml.e02_physics import gravity_forward_tf
from predict_unet import (
    calculate_metrics, save_comparison_figure, save_gravity_comparison,
    DENSITY_THRESHOLD, VOLUME_SHAPE,
)

CASES = ("E05A", "E05B", "E05C", "E05D", "E05E", "E05F")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def summarize(rows: list[dict], case: str) -> dict:
    if not rows:
        raise ValueError(f"No results for {case}")
    numeric = [key for key in rows[0] if key not in ("sample_id", "case", "empty_prediction")]
    return {
        "experiment": "E06", "case": case,
        "split": "test", "sample_count": len(rows),
        "density_threshold_gcm3": DENSITY_THRESHOLD,
        "empty_prediction_count": sum(bool(row["empty_prediction"]) for row in rows),
        "metrics": {
            name: {
                "mean": float(np.mean([float(row[name]) for row in rows])),
                "std": float(np.std([float(row[name]) for row in rows])),
            }
            for name in numeric
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--manifest", type=Path, default=Path("src/data/dataset_e06/manifest.csv"))
    p.add_argument("--data-root", type=Path, default=Path("outputs/ml/training/E06/data"))
    p.add_argument("--gravity-root", type=Path, default=Path("src/data/dataset_e06/forward_models"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/ml/predict/E06"))
    p.add_argument("--model", type=Path,
                   default=Path("outputs/ml/training/E06/checkpoints/best_unet3d.keras"))
    p.add_argument("--visualize-count", type=int, default=3,
                   help="Number of density and gravity PNG pairs per case (0=none).")
    p.add_argument("--plotly", action="store_true", help="Also export GIFs for visualized examples.")
    p.add_argument("--gif-frames", type=int, default=24)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing E06 evaluation outputs.")
    args = p.parse_args()
    if args.visualize_count < 0 or args.gif_frames < 2:
        p.error("visualize-count >= 0 and gif-frames >= 2 are required")

    manifest = resolve(args.manifest)
    data_root = resolve(args.data_root)
    gravity_root = resolve(args.gravity_root)
    output_root = resolve(args.output_root)
    model_path = resolve(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"E06 checkpoint missing: {model_path}")
    if not manifest.is_file():
        raise FileNotFoundError(f"E06 manifest missing: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    selected = dict.fromkeys(args.cases)
    grouped = defaultdict(list)
    for row in manifest_rows:
        if row["split"] == "test" and row["case"] in selected:
            grouped[row["case"]].append(row["sample_id"])
    for case in selected:
        if not grouped[case]:
            raise ValueError(f"No test samples for {case} in manifest")
        destination = output_root / case.replace("E05", "E06")
        if not args.overwrite and destination.exists() and any(destination.rglob("*.npz")):
            raise FileExistsError(f"Already evaluated {case}: {destination}; use --overwrite")

    print(f"Loading E06 checkpoint: {model_path}", flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    all_rows = []
    for case in selected:
        name = case.replace("E05", "E06")
        folder = output_root / name
        predictions = folder / "predictions" / "test"
        figures = folder / "figures" / "test"
        predictions.mkdir(parents=True, exist_ok=True)
        if args.visualize_count:
            figures.mkdir(parents=True, exist_ok=True)
        rows = []
        sample_ids = grouped[case]
        for index, sample_id in enumerate(sample_ids, 1):
            path = data_root / "test" / f"{sample_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            features, target = load_sample(path)
            prediction = model(features[None, ...], training=False).numpy()[0, ..., 0]
            truth = target[..., 0]
            if prediction.shape != VOLUME_SHAPE or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid prediction {sample_id}: {prediction.shape}")
            metrics = calculate_metrics(truth, prediction)
            row = {"sample_id": sample_id, "case": case, **metrics}
            rows.append(row)
            all_rows.append(row)
            with np.load(path) as archive:
                migration = np.asarray(archive["migration"], dtype=np.float32)
                scale = float(archive["migration_scale"])
            np.savez_compressed(predictions / f"{sample_id}.npz",
                                prediction=prediction.astype(np.float32),
                                density=truth.astype(np.float32),
                                migration=migration,
                                migration_input=features[..., 0],
                                migration_scale=np.float64(scale))
            if index <= args.visualize_count:
                save_comparison_figure(
                    truth, prediction, figures / f"{sample_id}_density_comparison.png", sample_id)
                gravity_path = gravity_root / "test" / f"{sample_id}.npz"
                with np.load(gravity_path) as f:
                    observed = np.asarray(f["gravity"], dtype=np.float32)
                recovered = gravity_forward_tf(
                    tf.convert_to_tensor(prediction[None, ..., None], dtype=tf.float32)
                ).numpy()[0, ..., 0]
                save_gravity_comparison(
                    observed, recovered, figures / f"{sample_id}_gravity_comparison.png", sample_id)
                if args.plotly:
                    from visualization.plotly_3d import plot_sample_3d
                    plot_sample_3d(truth, observed, figures / f"{sample_id}_3d.gif",
                                   reconstruction=prediction, make_gif=True,
                                   gif_frames=args.gif_frames)
            if index % 5 == 0 or index == len(sample_ids):
                print(f"{name}: {index}/{len(sample_ids)}", flush=True)

        pred_dir = folder / "predictions"
        with (pred_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows, name)
        (pred_dir / "test_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{name}: IoU={summary['metrics']['body_iou']['mean']:.4f} "
              f"Dice={summary['metrics']['body_dice']['mean']:.4f}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "all_test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    overall = summarize(all_rows, "all")
    overall["per_category"] = {
        case.replace("E05", "E06"): summarize(
            [r for r in all_rows if r["case"] == case], case.replace("E05", "E06"))
        for case in selected
    }
    (output_root / "aggregate_summary.json").write_text(json.dumps(overall, indent=2))
    print(f"E06 evaluation complete: {len(all_rows)} held-out samples; {output_root}", flush=True)


if __name__ == "__main__":
    main()
