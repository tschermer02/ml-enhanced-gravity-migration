"""Train E06 on the mixed-geology E06 dataset.

Unchanged from E05: model architecture, loss, learning rate, batch size,
seed, early stopping, learning-rate reduction, optimizer, and logging.
Changed: only training dataset (2000 mixed geologies + 120 validation).

Usage:
  python scripts/train_unet_e06.py --epochs 2 --limit 12 --output-root outputs/ml/training/E06_smoke
  python scripts/train_unet_e06.py --epochs 100 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

from gravity_migration_ml.ml.dataset import make_dataset, sample_paths
from gravity_migration_ml.ml.losses import balanced_density_mse
from gravity_migration_ml.ml.e03_losses import (
    soft_tversky_loss, TVERSKY_WEIGHT, TVERSKY_ALPHA, TVERSKY_BETA,
    DENSITY_THRESHOLD, OCCUPANCY_TEMPERATURE,
)
from gravity_migration_ml.ml.e05_losses import (
    e05_combined_loss, vertical_profile_loss, DEPTH_WEIGHT,
)
from gravity_migration_ml.ml.unet3d import build_unet3d

EXPERIMENT = "E06"
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
RANDOM_SEED = 42
DATA_ROOT = ROOT / "outputs" / "ml" / "training" / "E06" / "data"
OUTPUT_ROOT = ROOT / "outputs" / "ml" / "training" / EXPERIMENT


class BatchProgress(tf.keras.callbacks.Callback):
    """Print timing and original E05 loss terms."""
    def __init__(self, total_batches: int, total_epochs: int, interval: int = 10):
        super().__init__()
        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_start = time.perf_counter()
        print(f"\nStarting epoch {self.current_epoch}/{self.total_epochs}", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        completed = batch + 1
        if completed == 1 or completed % self.interval == 0 or completed == self.total_batches:
            logs = logs or {}
            print(f"Epoch {self.current_epoch}/{self.total_epochs} "
                  f"Batch {completed}/{self.total_batches} "
                  f"Loss={logs.get('loss', float('nan')):.6f} "
                  f"Elapsed={time.perf_counter() - self.epoch_start:.1f}s", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        tversky = logs.get("soft_tversky_loss", float("nan"))
        val_tversky = logs.get("val_soft_tversky_loss", float("nan"))
        depth = logs.get("vertical_profile_loss", float("nan"))
        val_depth = logs.get("val_vertical_profile_loss", float("nan"))
        print(
            f"Epoch {epoch+1}: loss={logs.get('loss', float('nan')):.6f}, "
            f"val_loss={logs.get('val_loss', float('nan')):.6f}, "
            f"balanced_mse={logs.get('balanced_density_mse', float('nan')):.6f}, "
            f"weighted_tversky={TVERSKY_WEIGHT*tversky:.6f}, "
            f"val_balanced_mse={logs.get('val_balanced_density_mse', float('nan')):.6f}, "
            f"val_weighted_tversky={TVERSKY_WEIGHT*val_tversky:.6f}, "
            f"weighted_depth={DEPTH_WEIGHT*depth:.6f}, "
            f"val_weighted_depth={DEPTH_WEIGHT*val_depth:.6f}", flush=True,
        )


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def plot_history(history: dict, figures_dir: Path) -> None:
    """Match E05 training loss and loss component visualizations."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, history["loss"], label="Training loss")
    ax.plot(epochs, history["val_loss"], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Combined loss", title="E06: Mixed Geology Training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "training_loss.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, weight, label in (
        ("balanced_density_mse", 1.0, "Balanced MSE"),
        ("soft_tversky_loss", TVERSKY_WEIGHT, "Weighted Tversky"),
        ("vertical_profile_loss", DEPTH_WEIGHT, "Weighted depth profile"),
    ):
        if name in history:
            ax.plot(epochs, weight * np.asarray(history[name]), label=label)
        if f"val_{name}" in history:
            ax.plot(epochs, weight * np.asarray(history[f"val_{name}"]),
                    label="Validation " + label, linestyle="--")
    ax.set(xlabel="Epoch", ylabel="Loss contribution",
           title="E06: Training and Validation Loss Components")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_components.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use first N per split, for a smoke test.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be >= 1")

    data_root = resolve(args.data_root)
    output_root = resolve(args.output_root)
    checkpoints = output_root / "checkpoints"
    logs = output_root / "logs"
    figures = output_root / "figures"
    for directory in (checkpoints, logs, figures):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite E06 artifacts in {directory}")
    train_paths = sample_paths(data_root, "train", args.limit)
    val_paths = sample_paths(data_root, "validation", args.limit)
    for folder in (checkpoints, logs, figures):
        folder.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    train_data = make_dataset(data_root, "train", batch_size=args.batch_size,
                              limit=args.limit, seed=RANDOM_SEED)
    val_data = make_dataset(data_root, "validation", batch_size=args.batch_size,
                            limit=args.limit, seed=RANDOM_SEED)
    model = build_unet3d()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=e05_combined_loss,
        metrics=[balanced_density_mse, soft_tversky_loss, vertical_profile_loss],
        jit_compile=False,
    )
    model.summary()
    checkpoint_path = checkpoints / "best_unet3d.keras"
    history_path = logs / "training_history.csv"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(history_path),
        BatchProgress((len(train_paths) + args.batch_size - 1) // args.batch_size, args.epochs),
    ]
    config = {
        "experiment": EXPERIMENT,
        "description": "E05 network/loss/hyperparameters retrained from scratch on mixed geology",
        "comparison_experiment": "E05",
        "changed_variable": "dataset (six mixed geology families)",
        "architecture": "E01 3D U-Net", "input_shape": [24, 64, 64, 1],
        "loss": "balanced_mse_plus_soft_tversky_plus_vertical_profile",
        "balanced_mse_weight": 1.0, "tversky_weight": TVERSKY_WEIGHT,
        "tversky_alpha": TVERSKY_ALPHA, "tversky_beta": TVERSKY_BETA,
        "depth_loss_weight": DEPTH_WEIGHT,
        "density_threshold_gcm3": DENSITY_THRESHOLD,
        "occupancy_temperature_gcm3": OCCUPANCY_TEMPERATURE,
        "learning_rate": LEARNING_RATE, "batch_size": args.batch_size,
        "max_epochs": args.epochs, "seed": RANDOM_SEED,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE, "jit_compile": False,
        "train_count": len(train_paths), "validation_count": len(val_paths),
        "data_root": str(data_root), "checkpoint": str(checkpoint_path),
    }
    (logs / "training_config.json").write_text(json.dumps(config, indent=2))
    print("Starting E06 training...", flush=True)
    history = model.fit(train_data, validation_data=val_data, epochs=args.epochs,
                        callbacks=callbacks, verbose=0)
    plot_history(history.history, figures)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"E06 COMPLETE: best epoch {best_epoch}, "
          f"val_loss={np.min(history.history['val_loss']):.6f}, "
          f"checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
