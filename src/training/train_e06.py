"""Train E06 on the mixed-geology E06 dataset.

Unchanged from E05: model architecture, loss, learning rate, batch size,
seed, early stopping, learning-rate reduction, optimizer, and logging.
Changed: negative E05D targets and signed tanh head / abs-occupancy losses.

Usage:
  python scripts/train_unet_e06_signed.py --epochs 2 --limit 12 --output-root outputs/ml/training/E06_signed_smoke
  python scripts/train_unet_e06_signed.py --epochs 100 --batch-size 2
"""
from __future__ import annotations
# BEGIN UNIFIED TRAINING IMPORTS

from src.data.dataset import make_dataset, sample_paths
from src.models.unet3d import build_unet3d
from src.losses.losses import (
    ordinary_mse, balanced_density_mse, balanced_huber_loss,
    soft_tversky_loss, vertical_profile_loss, make_loss,
)

TVERSKY_WEIGHT = 0.1
TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3
DENSITY_THRESHOLD = 0.1
OCCUPANCY_TEMPERATURE = 0.025
DEPTH_WEIGHT = 0.02
THRESHOLD = DENSITY_THRESHOLD
TEMP = OCCUPANCY_TEMPERATURE

e03_combined_loss = make_loss(weights={"balanced_mse": 1, "tversky": TVERSKY_WEIGHT})
e05_combined_loss = make_loss(weights={"balanced_mse": 1, "tversky": TVERSKY_WEIGHT, "depth": DEPTH_WEIGHT})
signed_e06_loss = make_loss(weights={"balanced_mse": 1, "tversky": TVERSKY_WEIGHT, "depth": DEPTH_WEIGHT}, signed=True)

def signed_balanced_density_mse(y_true, y_pred):
    return balanced_density_mse(y_true, y_pred, signed=True)

def signed_tversky(y_true, y_pred):
    return soft_tversky_loss(y_true, y_pred, signed=True, subtract_baseline=True)

def signed_vertical_profile(y_true, y_pred):
    return vertical_profile_loss(y_true, y_pred, signed=True)

# END UNIFIED TRAINING IMPORTS


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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


EXPERIMENT = "E06_signed"
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
RANDOM_SEED = 42
DATA_ROOT = ROOT / "outputs" / "ml" / "training" / "E06_signed" / "data"
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
        tversky = logs.get("signed_tversky", float("nan"))
        val_tversky = logs.get("val_signed_tversky", float("nan"))
        depth = logs.get("signed_vertical_profile", float("nan"))
        val_depth = logs.get("val_signed_vertical_profile", float("nan"))
        print(
            f"Epoch {epoch+1}: loss={logs.get('loss', float('nan')):.6f}, "
            f"val_loss={logs.get('val_loss', float('nan')):.6f}, "
            f"balanced_mse={logs.get('signed_balanced_density_mse', float('nan')):.6f}, "
            f"weighted_tversky={TVERSKY_WEIGHT*tversky:.6f}, "
            f"val_balanced_mse={logs.get('val_signed_balanced_density_mse', float('nan')):.6f}, "
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
        ("signed_balanced_density_mse", 1.0, "Balanced MSE"),
        ("signed_tversky", TVERSKY_WEIGHT, "Weighted Tversky"),
        ("signed_vertical_profile", DEPTH_WEIGHT, "Weighted depth profile"),
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
    train_paths = sorted((data_root / "train").glob("*.npz"))[:args.limit]
    val_paths = sorted((data_root / "validation").glob("*.npz"))[:args.limit]
    for folder in (checkpoints, logs, figures):
        folder.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    train_data = make_dataset(data_root, "train", batch_size=args.batch_size,
                              limit=args.limit, seed=RANDOM_SEED)
    val_data = make_dataset(data_root, "validation", batch_size=args.batch_size,
                            limit=args.limit, seed=RANDOM_SEED)
    # Reuse E05 feature extractor, replace sigmoid head with signed tanh output.
    original = build_unet3d()
    signed_head = tf.keras.layers.Conv3D(1, 1, activation="tanh", name="signed_density_prediction")(
        original.get_layer("decoder1_conv2").output
    )
    model = tf.keras.Model(original.input, signed_head, name="e06_signed_unet3d")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=signed_e06_loss,
        metrics=[signed_balanced_density_mse, signed_tversky, signed_vertical_profile],
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
        "description": "Signed E06: same mixed geologies but realistic negative E05D; new signed output/loss",
        "comparison_experiment": "E05",
        "changed_variable": "negative salt-dome E05D; signed output and occupancy-aware losses",
        "architecture": "E01 3D U-Net with tanh signed-density head", "input_shape": [24, 64, 64, 1],
        "loss": "signed_balanced_mse_plus_signed_tversky_plus_signed_vertical_profile",
        "balanced_mse_weight": 1.0, "tversky_weight": TVERSKY_WEIGHT,
        "tversky_alpha": 0.7, "tversky_beta": 0.3,
        "depth_loss_weight": DEPTH_WEIGHT,
        "density_threshold_gcm3": THRESHOLD,
        "occupancy_temperature_gcm3": TEMP,
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
