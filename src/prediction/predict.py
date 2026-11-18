"""Run a trained 3D U-Net on saved migration volumes.

Legacy usage (unchanged):
    python -m src.prediction.predict --name E08 --split test
    python -m src.prediction.predict --name E08 --model path/to/model.keras

E11 usage (dataset automatically selected by experiment):
    python -m src.prediction.predict --name E11A --split test
    python -m src.prediction.predict --name E11E --split test --limit 5

Prediction uses the *stored* migration_input exactly as training did. It does
not recompute migration, normalize a second time, or perform gravity physics.
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
E11_DATASETS = {
    "E11BASE": "D11_BASE",
    "E11A": "D11_INTRUSIVE",
    "E11B": "D11_HEIGHT",
    "E11C": "D11_BASE",
    "E11D": "D11_COMBINED",
    "E11E": "D11_COMBINED",
}
E11_CELL_DIMS_M = (125.0, 125.0, 1000.0 / 24)


def sample_paths(data_root: Path, split: str, samples=None, limit=None):
    """Select samples deterministically, optionally by filename/sample ID."""
    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}")
    folder = Path(data_root) / split
    paths = sorted(folder.rglob("*.npz"))
    if samples:
        wanted = {str(v).removesuffix(".npz") for v in samples}
        paths = [p for p in paths if p.stem in wanted or
                 str(p.relative_to(folder)).removesuffix(".npz") in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No samples found in {folder}")
    return paths


def load_prediction_input(path: Path):
    """Load the saved migration and preserve source metadata for evaluation.

    Copy all original NPZ arrays rather than relying on a fixed metadata list.
    This preserves E11 cell sizes and observation height, along with original
    legacy fields, while preventing a duplicate migration_input on save.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if "migration_input" not in archive:
            raise KeyError(f"{path}: missing migration_input; generate physics first")
        inputs = np.asarray(archive["migration_input"], dtype=np.float32)
        arrays = {key: archive[key].copy() for key in archive.files
                  if key not in ("migration_input", "prediction")}
    if inputs.shape != VOLUME_SHAPE or not np.isfinite(inputs).all():
        raise ValueError(f"{path}: invalid migration_input shape or values: {inputs.shape}")
    if "density" in arrays and arrays["density"].shape != VOLUME_SHAPE:
        raise ValueError(f"{path}: unexpected density shape {arrays['density'].shape}")
    return inputs, arrays


def validate_e11_sample(arrays: dict, path: Path):
    """Catch mismatched geometry/height before labeling output as E11."""
    for key, expected in zip(("dx_m", "dy_m", "dz_m"), E11_CELL_DIMS_M):
        if key not in arrays:
            raise ValueError(f"{path}: E11 sample missing {key}")
        value = float(np.asarray(arrays[key]).item())
        if not np.isfinite(value) or not np.isclose(value, expected, rtol=1e-6):
            raise ValueError(f"{path}: {key}={value}, expected {expected}")
    if "observation_height_m" not in arrays:
        raise ValueError(f"{path}: E11 sample missing observation_height_m")
    height = float(np.asarray(arrays["observation_height_m"]).item())
    if not np.isfinite(height) or height < 0:
        raise ValueError(f"{path}: invalid observation height {height}")
    component = arrays.get("observation_component")
    if component is None or str(np.asarray(component).item()).lower() != "gzz":
        raise ValueError(f"{path}: E11 requires observation_component='GZZ'")


def predict_one(model, input_volume):
    """Predict one density volume without compiling the trained checkpoint."""
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
                    limit=None, overwrite=False, progress_every=25,
                    validate_e11=False):
    """Write one prediction NPZ per sample without calculating metrics."""
    data_root = Path(data_root)
    output_root = Path(output_root)
    paths = sample_paths(data_root, split, samples=samples, limit=limit)
    source_split = data_root / split
    target_paths = []
    for path in paths:
        parts = path.relative_to(source_split).parts
        # Preserve legacy flat layout and geometry/noise-specific subfolders.
        target = output_root.joinpath(*parts[:-1], "predictions", split, path.name)
        target_paths.append(target)
    existing = [p for p in target_paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {existing[0]} (use --overwrite)")

    records = []
    for index, (path, target) in enumerate(zip(paths, target_paths), 1):
        input_volume, original = load_prediction_input(path)
        if validate_e11:
            validate_e11_sample(original, path)
        prediction = predict_one(model, input_volume)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, prediction=prediction,
                            migration_input=input_volume, **original)
        records.append({"sample_id": path.stem, "source": str(path),
                        "prediction": str(target)})
        if index % progress_every == 0 or index == len(paths):
            print(f"{split}: {index}/{len(paths)}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{split}_prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "source", "prediction"))
        writer.writeheader()
        writer.writerows(records)
    print(f"Finished predicting {split} dataset.", flush=True)
    return records


def _load_json_if_exists(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Training run name, e.g. E08D or E11A")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name (E11 experiments auto-map to D11_...)")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset or E11_DATASETS.get(args.name, args.name)
    data_root = args.data_root or ROOT / "outputs" / "ml" / "training" / dataset_name / "data"
    model_path = args.model or ROOT / "outputs" / "ml" / "training" / args.name / "checkpoints" / "best_unet3d.keras"
    output_root = args.output_root or ROOT / "outputs" / "ml" / "predict" / args.name
    # Resolve paths consistently from the repository root, regardless of CWD.
    data_root = data_root if data_root.is_absolute() else ROOT / data_root
    model_path = model_path if model_path.is_absolute() else ROOT / model_path
    output_root = output_root if output_root.is_absolute() else ROOT / output_root
    is_e11 = args.name in E11_DATASETS

    training_config = _load_json_if_exists(model_path.parent.parent / "logs" / "training_config.json")
    dataset_config = _load_json_if_exists(data_root / "config.json")
    if is_e11:
        if dataset_config is None or dataset_config.get("grid") != "e11":
            raise ValueError("E11 prediction requires a dataset with grid='e11' in config.json")
        if dataset_config.get("component", "").lower() != "gzz":
            raise ValueError("E11 prediction requires a Gzz dataset")
        for key, expected in zip(("dx_m", "dy_m", "dz_m"), E11_CELL_DIMS_M):
            if not np.isclose(float(dataset_config.get(key, -1)), expected, rtol=1e-6):
                raise ValueError(f"Dataset configuration has wrong {key}")
        if training_config is not None:
            for key in ("dx_m", "dy_m", "dz_m"):
                if key in training_config and not np.isclose(
                    float(training_config[key]), float(dataset_config[key]), rtol=1e-6
                ):
                    raise ValueError(f"Checkpoint and prediction dataset disagree on {key}")

    import tensorflow as tf  # Lazy import so helper tests need only NumPy.
    model = tf.keras.models.load_model(model_path, compile=False)
    records = run_predictions(
        model, data_root, output_root, split=args.split,
        samples=args.samples, limit=args.limit, overwrite=args.overwrite,
        validate_e11=is_e11,
    )
    meta = {"run_name": args.name, "dataset_name": dataset_name,
            "model_path": str(model_path), "data_root": str(data_root),
            "split": args.split, "sample_count": len(records)}
    if is_e11:
        meta.update({"grid": "e11", "model_extent_m": [8000.0, 8000.0, 1000.0],
                     "dx_m": E11_CELL_DIMS_M[0], "dy_m": E11_CELL_DIMS_M[1],
                     "dz_m": E11_CELL_DIMS_M[2],
                     "observation_heights_m": dataset_config.get("observation_heights_m"),
                     "normalization": "stored_migration_input"})
    (output_root / f"{args.split}_prediction_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
