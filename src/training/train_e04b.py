"""Train E04B: identical E04 network/loss/hyperparameters, new 2-3-body dataset.

E04B dataset setup (from repository root):
    python src/data/generate_dataset_multibody.py --train 1000 --validation 100 --test 100 --min-bodies 2 --max-bodies 3 --gap-cells 0 --seed 42 --output-root src/data/dataset_multibody_e04b/synthetic_models

Training:
    python scripts/train_unet_e04b.py --epochs 100 --batch-size 2

The existing generator uses half-open intervals. A gap of zero allows touching
faces/edges/corners and overlap in individual coordinate projections, but never
overlapping occupied 3D cells.
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


# Exact E04 training settings; only the dataset path is different.
EXPERIMENT = "E04B"
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
RANDOM_SEED = 42

DATA_ROOT = ROOT / "outputs" / "ml" / "training" / "E04B" / "data"
OUTPUT_ROOT = ROOT / "outputs" / "ml" / "training" / EXPERIMENT


class BatchProgress(tf.keras.callbacks.Callback):
    """Display batch timing and the same loss components as E04."""

    def __init__(self, total_batches: int, total_epochs: int, interval: int = 10):
        super().__init__()
        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_start = time.perf_counter()
        print(f"\\nStarting epoch {self.current_epoch}/{self.total_epochs}", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        completed = batch + 1
        if completed == 1 or completed % self.interval == 0 or completed == self.total_batches:
            logs = logs or {}
            print(
                f"Epoch {self.current_epoch}/{self.total_epochs} "
                f"Batch {completed}/{self.total_batches} "
                f"Loss={logs.get('loss', float('nan')):.6f} "
                f"Elapsed={time.perf_counter() - self.epoch_start:.1f}s",
                flush=True,
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        density = logs.get("balanced_density_mse", float("nan"))
        tversky = logs.get("soft_tversky_loss", float("nan"))
        val_density = logs.get("val_balanced_density_mse", float("nan"))
        val_tversky = logs.get("val_soft_tversky_loss", float("nan"))
        print(
            f"Epoch {epoch+1}: loss={logs.get('loss', float('nan')):.6f}, "
            f"val_loss={logs.get('val_loss', float('nan')):.6f}, "
            f"balanced_mse={density:.6f}, "
            f"soft_tversky={tversky:.6f}, "
            f"weighted_tversky={TVERSKY_WEIGHT * tversky:.6f}, "
            f"val_balanced_mse={val_density:.6f}, "
            f"val_weighted_tversky={TVERSKY_WEIGHT * val_tversky:.6f}",
            flush=True,
        )


def resolve_path(path: Path) -> Path:
    """Resolve relative paths from repository root, not current working directory."""
    return path if path.is_absolute() else ROOT / path


def plot_history(history: dict[str, list[float]], figures_dir: Path) -> None:
    """Write E04-compatible training and loss-component figures."""
    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, history["loss"], label="Training loss")
    ax.plot(epochs, history["val_loss"], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Combined loss",
           title="E04B: Multi-body Balanced MSE + Soft Tversky")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "training_loss.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    if "balanced_density_mse" in history:
        ax.plot(epochs, history["balanced_density_mse"], label="Balanced MSE")
    if "soft_tversky_loss" in history:
        ax.plot(
            epochs, TVERSKY_WEIGHT * np.asarray(history["soft_tversky_loss"]),
            label=f"Weighted Tversky ({TVERSKY_WEIGHT}×)",
        )
    if "val_balanced_density_mse" in history:
        ax.plot(epochs, history["val_balanced_density_mse"],
                label="Validation balanced MSE", linestyle="--")
    if "val_soft_tversky_loss" in history:
        ax.plot(
            epochs, TVERSKY_WEIGHT * np.asarray(history["val_soft_tversky_loss"]),
            label="Validation weighted Tversky", linestyle="--",
        )
    ax.set(xlabel="Epoch", ylabel="Loss contribution",
           title="E04B: Training and Validation Loss Components")
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
                        help="First N samples per split (for smoke testing).")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root)
    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "logs"
    figures_dir = output_root / "figures"

    # Dataset may live at output_root/data, so only protect training artifacts.
    for directory in (checkpoints_dir, logs_dir, figures_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite experiment artifacts: {directory}\\n"
                "Choose a different --output-root for another training run."
            )

    train_paths = sample_paths(data_root, "train", args.limit)
    validation_paths = sample_paths(data_root, "validation", args.limit)
    train_count = len(train_paths)
    validation_count = len(validation_paths)
    print(f"Experiment: {EXPERIMENT}", flush=True)
    print(f"Data: {data_root}", flush=True)
    print(f"Training samples: {train_count}", flush=True)
    print(f"Validation samples: {validation_count}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)

    for directory in (checkpoints_dir, logs_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_data = make_dataset(
        data_root, "train", batch_size=args.batch_size,
        limit=args.limit, seed=RANDOM_SEED,
    )
    validation_data = make_dataset(
        data_root, "validation", batch_size=args.batch_size,
        limit=args.limit, seed=RANDOM_SEED,
    )

    # Intentionally unchanged E04 model/loss/optimizer.
    model = build_unet3d()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=e03_combined_loss,
        metrics=[balanced_density_mse, soft_tversky_loss],
        jit_compile=False,
    )
    model.summary()

    checkpoint_path = checkpoints_dir / "best_unet3d.keras"
    history_path = logs_dir / "training_history.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(history_path),
        BatchProgress(
            total_batches=(train_count + args.batch_size - 1) // args.batch_size,
            total_epochs=args.epochs,
        ),
    ]

    config = {
        "experiment": EXPERIMENT,
        "description": "E04 architecture, loss, and training settings with 2-3 touching-allowed prisms.",
        "comparison_experiment": "E04",
        "changed_variable": "dataset only: 2-3 bodies, 0-cell gap instead of 1-3 bodies, 2-cell gap",
        "architecture": "E01 3D U-Net",
        "input_shape": [24, 64, 64, 1],
        "loss": "balanced_mse_plus_soft_tversky",
        "balanced_mse_weight": 1.0,
        "tversky_weight": TVERSKY_WEIGHT,
        "tversky_alpha": TVERSKY_ALPHA,
        "tversky_beta": TVERSKY_BETA,
        "density_threshold_gcm3": DENSITY_THRESHOLD,
        "occupancy_mapping": "sigmoid",
        "occupancy_temperature_gcm3": OCCUPANCY_TEMPERATURE,
        "learning_rate": LEARNING_RATE,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "seed": RANDOM_SEED,
        "jit_compile": False,
        "train_count": train_count,
        "validation_count": validation_count,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "checkpoint": str(checkpoint_path),
    }
    (logs_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\\nStarting E04B training...\\n", flush=True)
    history = model.fit(
        train_data, validation_data=validation_data,
        epochs=args.epochs, callbacks=callbacks, verbose=0,
    )
    plot_history(history.history, figures_dir)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))

    print("\\n================================", flush=True)
    print("E04B TRAINING COMPLETE", flush=True)
    print("================================", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best validation loss: {best_val_loss:.6f}", flush=True)
    print(f"Best model: {checkpoint_path}", flush=True)
    print(f"Training history: {history_path}", flush=True)
    print(f"Figures: {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
