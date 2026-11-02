"""Run a trained 3D network on a saved gravity-migration dataset.

Prediction is deliberately separate from metrics and plotting.
Run from the repository root:
    python -m src.prediction.predict --name E08 --split test
    python -m src.prediction.predict --name E08 --split test --limit 5
    python -m src.prediction.predict --name E08 --model path/to/model.keras

The dataset may contain positive or signed density, Gz or Gzz. The network
receives the stored normalized migration_input, not raw gravity data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
VOLUME_SHAPE = (24, 64, 64)


def sample_paths(data_root: Path, split: str, samples=None, limit=None):
    """Select samples deterministically, optionally by filename or sample ID."""
    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}")
    folder = data_root / split
    paths = sorted(folder.rglob("*.npz"))
    if samples:
        wanted = {str(v).removesuffix(".npz") for v in samples}
        paths = [p for p in paths if p.stem in wanted or str(p.relative_to(folder)).removesuffix(".npz") in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No samples found in {folder}")
    return paths


def load_prediction_input(path: Path):
    """Load normalized migration and any metadata needed downstream."""
    with np.load(path, allow_pickle=False) as archive:
        if "migration_input" not in archive:
            raise KeyError(f"{path}: missing migration_input; generate physics first")
        inputs = np.asarray(archive["migration_input"], dtype=np.float32)
        arrays = {}
        for name in ("density", "gravity", "gravity_clean", "migration"):
            if name in archive:
                arrays[name] = np.asarray(archive[name], dtype=np.float32)
        for name in ("migration_scale", "noise_percent", "geometry", "observed_component", "observation_component", "observation_unit", "model_id"):
            if name in archive:
                arrays[name] = archive[name].copy()
    if inputs.shape != VOLUME_SHAPE or not np.isfinite(inputs).all():
        raise ValueError(f"{path}: invalid migration_input shape or values: {inputs.shape}")
    if "density" in arrays and arrays["density"].shape != VOLUME_SHAPE:
        raise ValueError(f"{path}: unexpected density shape {arrays['density'].shape}")
    return inputs, arrays


def predict_one(model, input_volume):
    """Predict one density volume without calling model.compile()."""
    result = model(input_volume[None, ..., None], training=False)
    if hasattr(result, "numpy"):
        result = result.numpy()
    prediction = np.asarray(result, dtype=np.float32)
    if prediction.shape != (1, *VOLUME_SHAPE, 1):
        raise ValueError(f"Model returned shape {prediction.shape}, expected (1,24,64,64,1)")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinity")
    return prediction[0, ..., 0]


def run_predictions(model, data_root, output_root, split="test", samples=None,
                    limit=None, overwrite=False, progress_every=25):
    """Write one NPZ per sample, without calculating evaluation metrics."""
    data_root = Path(data_root)
    output_root = Path(output_root)
    paths = sample_paths(data_root, split, samples=samples, limit=limit)
    source_split = data_root / split
    target_paths = []
    for path in paths:
        parts = path.relative_to(source_split).parts
        # New layout: experiment/geometry[/noise_NN]/predictions/split/sample.
        # Keep legacy flat-data layout compatible.
        target = output_root.joinpath(*parts[:-1], "predictions", split, path.name)
        target_paths.append(target)
    existing = [p for p in target_paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {existing[0]} (use --overwrite)")
    for target in target_paths:
        target.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (path, target) in enumerate(zip(paths, target_paths), 1):
        input_volume, original = load_prediction_input(path)
        prediction = predict_one(model, input_volume)
        np.savez_compressed(target, prediction=prediction,
                            migration_input=input_volume, **original)
        records.append({"sample_id": path.stem, "source": str(path),
                        "prediction": str(target)})
        if index % progress_every == 0 or index == len(paths):
            print(f"{split}: {index}/{len(paths)}", flush=True)
    manifest_path = output_root / f"{split}_prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "source", "prediction"))
        writer.writeheader()
        writer.writerows(records)
    print(f"Finished predicting {split} dataset.", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Output run name, e.g. E08")
    parser.add_argument("--dataset", default=None, help="Dataset experiment name (default: --name)")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dataset_name = args.dataset or args.name
    data_root = args.data_root or ROOT / "outputs" / "ml" / "training" / dataset_name / "data"
    model_path = args.model or ROOT / "outputs" / "ml" / "training" / args.name / "checkpoints" / "best_unet3d.keras"
    output_root = args.output_root or ROOT / "outputs" / "ml" / "predict" / args.name
    import tensorflow as tf  # Lazy import so helper tests need only NumPy.
    model = tf.keras.models.load_model(model_path, compile=False)
    records = run_predictions(model, data_root, output_root, split=args.split,
                              samples=args.samples, limit=args.limit,
                              overwrite=args.overwrite)
    meta = {"run_name": args.name, "dataset_name": dataset_name,
            "model_path": str(model_path), "data_root": str(data_root),
            "split": args.split, "sample_count": len(records)}
    (Path(output_root) / f"{args.split}_prediction_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
