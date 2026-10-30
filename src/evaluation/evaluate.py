"""Evaluate a saved Keras density inversion model against a dataset split.

Example (from the repository root):
    python -m src.evaluation.evaluate --name TEST01 --split test
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .metrics import calculate_metrics, summarize, grouped_summary

ROOT = Path(__file__).resolve().parents[2]


def evaluate(data_root, model_path, output_root, split="test", *, threshold=0.1,
             cell_size_m=10.0, limit=None, overwrite=False):
    # Lazy import lets pure NumPy metrics be used without importing TensorFlow.
    import tensorflow as tf
    data_root, model_path, output_root = map(Path, (data_root, model_path, output_root))
    files = sorted((data_root / split).rglob("*.npz"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No {split} samples in {data_root}")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    marker = output_root / f"{split}_summary.json"
    if marker.exists() and not overwrite:
        raise FileExistsError(f"Evaluation exists at {marker}; use --overwrite")

    model = None  # Only load the checkpoint if a prediction is missing.
    rows = []
    for i, path in enumerate(files, 1):
        with np.load(path, allow_pickle=False) as f:
            x = np.asarray(f["migration_input"], dtype=np.float32)
            truth = np.asarray(f["density"], dtype=np.float32)
            geometry = str(f["geometry"].item()) if "geometry" in f else "unknown"
            noise = float(f["noise_percent"].item()) if "noise_percent" in f else 0.0
            migration = np.asarray(f["migration"], dtype=np.float32) if "migration" in f else None
        if x.shape != truth.shape or x.ndim != 3:
            raise ValueError(f"Unexpected sample shape in {path}")
        relative = path.relative_to(data_root / split)
        pred_path = output_root.joinpath(*relative.parts[:-1], "predictions", split, path.name)
        if pred_path.exists():
            with np.load(pred_path, allow_pickle=False) as saved:
                pred = np.asarray(saved["prediction"], dtype=np.float32).squeeze()
        else:
            if model is None:
                model = tf.keras.models.load_model(model_path, compile=False)
            pred = np.asarray(model(x[None, ..., None], training=False).numpy()).squeeze()
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with np.load(path, allow_pickle=False) as sample:
                payload = {key: sample[key] for key in sample.files}
            np.savez_compressed(pred_path, **payload, prediction=pred.astype(np.float32))
        metrics = calculate_metrics(truth, pred, threshold, cell_size_m)
        rows.append({"sample_id": path.stem, "split": split, "geometry": geometry,
                     "noise_percent": noise, **metrics})
        # Preserve existing prediction files (including gravity metadata).
        if i % 25 == 0 or i == len(files):
            print(f"{split}: {i}/{len(files)}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f"{split}_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"split": split, "overall": summarize(rows),
               "by_geometry_and_noise": grouped_summary(rows),
               "model_path": str(model_path), "data_root": str(data_root),
               "threshold_gcm3": threshold, "cell_size_m": cell_size_m}
    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for geom in sorted({row["geometry"] for row in rows}):
        group_rows = [row for row in rows if row["geometry"] == geom]
        geometry_dir = output_root / geom
        geometry_dir.mkdir(parents=True, exist_ok=True)
        with (geometry_dir / f"{split}_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(group_rows)
        (geometry_dir / f"{split}_summary.json").write_text(
            json.dumps({"geometry": geom, "overall": summarize(group_rows),
                        "by_geometry_and_noise": grouped_summary(group_rows)}, indent=2),
            encoding="utf-8")
    print(f"Finished evaluating {split} dataset: {output_root}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Run name used only for input/output paths")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--cell-size", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    training = ROOT / "outputs" / "ml" / "training" / args.name
    evaluate(args.data_root or training / "data",
             args.model or training / "checkpoints" / "best_unet3d.keras",
             args.output_root or ROOT / "outputs" / "ml" / "predict" / args.name,
             args.split, threshold=args.threshold, cell_size_m=args.cell_size,
             limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
