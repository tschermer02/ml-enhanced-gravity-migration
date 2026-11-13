
"""E10D inference: raw D06C migration with fixed training-derived scaling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHAPE = (24, 64, 64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--training-root",
        type=Path,
        default=ROOT / "outputs/ml/training/E10D",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/ml/predict/E10D",
    )
    p.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from src.data.dataset import sample_paths

    files = sample_paths(args.data_root, args.split, args.limit)

    training_root = args.training_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    # Load the EXACT scaling factor used during training.
    config_path = training_root / "logs/training_config.json"

    with open(config_path) as f:
        config = json.load(f)

    scale = float(config["migration_scale"])

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid fixed migration scale")

    print(f"Using fixed training scale: {scale:.8g}")

    import tensorflow as tf

    model_path = (
        training_root / "checkpoints/best_unet3d.keras"
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    if tuple(model.input_shape[1:]) != (*SHAPE, 1):
        raise ValueError(
            f"Unexpected model input shape: {model.input_shape}"
        )

    records = []

    for i, path in enumerate(files, 1):

        relative = path.relative_to(
            args.data_root / args.split
        )

        target = output_root.joinpath(
            *relative.parts[:-1],
            "predictions",
            args.split,
            relative.name,
        )

        if target.exists() and not args.overwrite:
            raise FileExistsError(
                f"{target} already exists. Use --overwrite."
            )

        # Load ORIGINAL raw migration.
        with np.load(path, allow_pickle=False) as f:
            raw = np.asarray(
                f["migration"], dtype=np.float32
            )

            original = {
                k: f[k].copy()
                for k in f.files
                if k not in ("migration_input", "migration")
            }

        if raw.shape != SHAPE:
            raise ValueError(
                f"Unexpected migration shape: {raw.shape}"
            )

        if not np.isfinite(raw).all():
            raise ValueError("Migration contains NaN or Inf")

        # Apply SAME fixed scaling as training.
        network_input = (raw / scale).astype(np.float32)

        # Predict density.
        prediction = np.asarray(
            model(
                network_input[None, ..., None],
                training=False,
            )
        )[0, ..., 0]

        if prediction.shape != SHAPE:
            raise ValueError(
                f"Unexpected prediction shape: {prediction.shape}"
            )

        if not np.isfinite(prediction).all():
            raise ValueError("Prediction contains NaN or Inf")

        # Save all original arrays and exact network input.
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            target,
            **original,
            migration=raw,
            migration_input=network_input,
            prediction=prediction.astype(np.float32),
            fixed_migration_scale=np.float64(scale),
        )

        records.append({
            "source": str(path),
            "output": str(target),
        })

        if i % 25 == 0 or i == len(files):
            print(
                f"{args.split}: {i}/{len(files)}",
                flush=True,
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_root / f"{args.split}_prediction_manifest.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["source", "output"],
        )

        writer.writeheader()
        writer.writerows(records)

    config_output = {
        "experiment": "E10D",
        "scale": scale,
        "source": "raw migration",
        "transform": "network_input = raw migration / fixed scale",
        "training_root": str(training_root),
        "dataset": str(args.data_root),
        "split": args.split,
    }

    with open(
        output_root / f"{args.split}_prediction_config.json",
        "w",
    ) as f:
        json.dump(config_output, f, indent=2)

    print("Predictions saved to:", output_root)


if __name__ == "__main__":
    main()
