"""Train the baseline 3D U-Net on normalized migration volumes."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gravity_migration_ml.ml.dataset import make_dataset  # noqa: E402
from gravity_migration_ml.ml.losses import balanced_density_mse, ordinary_mse  # noqa: E402
from gravity_migration_ml.ml.unet3d import build_unet3d  # noqa: E402

# Baseline settings; command-line arguments can override the run length and batch size.
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
RANDOM_SEED = 42
OUTPUT_ROOT = ROOT / "outputs" / "ml" / "training" / "E01"
DATA_ROOT = OUTPUT_ROOT / "data"


class EpochLossPrinter(tf.keras.callbacks.Callback):
    """Print the current training and validation losses after every epoch."""

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        print(
            f"Epoch {epoch + 1}: loss={logs.get('loss', float('nan')):.6g}, "
            f"val_loss={logs.get('val_loss', float('nan')):.6g}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N samples from each split for a short smoke test.")
    parser.add_argument("--loss", choices=("balanced", "mse"), default="balanced")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "logs"
    figures_dir = output_root / "figures"
    for directory in (checkpoints_dir, logs_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    train_data = make_dataset(
        data_root, "train", args.batch_size, limit=args.limit, seed=RANDOM_SEED,
    )
    validation_data = make_dataset(
        data_root, "validation", args.batch_size, limit=args.limit, seed=RANDOM_SEED,
    )

    model = build_unet3d()
    loss_function = balanced_density_mse if args.loss == "balanced" else ordinary_mse
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_function,
    )
    checkpoint_path = checkpoints_dir / "best_unet3d.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(logs_dir / "training_history.csv"),
        EpochLossPrinter(),
    ]

    config = {
        "experiment": "E01",
        "input_shape": [24, 64, 64, 1],
        "learning_rate": LEARNING_RATE,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "loss": args.loss,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "seed": RANDOM_SEED,
        "train_limit": args.limit,
        "data_root": str(data_root),
        "checkpoint": str(checkpoint_path),
    }
    (logs_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(epochs, history.history["loss"], label="Training loss")
    axis.plot(epochs, history.history["val_loss"], label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="Loss", title="3D U-Net training history")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(figures_dir / "training_loss.png", dpi=160)
    plt.close(fig)
    print(f"Best model: {checkpoint_path}")


if __name__ == "__main__":
    main()
