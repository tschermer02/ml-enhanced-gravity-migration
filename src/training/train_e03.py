
"""
Train E03: Balanced MSE + Soft Tversky loss.

E01: Balanced density MSE
E02: Balanced Huber
E03: Balanced density MSE + 0.1 Soft Tversky

The architecture, dataset, preprocessing, and training
hyperparameters remain unchanged from E01.

E03 adds geometry supervision to improve body overlap
and reduce false-positive density predictions.
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


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))






# ============================================================
# 2. EXPERIMENT PARAMETERS
# ============================================================

EXPERIMENT = "E03"

LEARNING_RATE = 1e-3
BATCH_SIZE = 2
MAX_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5

RANDOM_SEED = 42

# Reuse exactly the same preprocessed dataset as E01/E02.
DATA_ROOT = (
    ROOT / "outputs" / "ml" / "training" / "E01" / "data"
)

# Keep E03 results separate from previous experiments.
OUTPUT_ROOT = (
    ROOT / "outputs" / "ml" / "training" / EXPERIMENT
)


# ============================================================
# 3. TRAINING PROGRESS CALLBACK
# ============================================================

class BatchProgress(tf.keras.callbacks.Callback):
    """Print batch progress and epoch-level loss components."""

    def __init__(
        self,
        total_batches: int,
        total_epochs: int,
        interval: int = 10,
    ):
        super().__init__()

        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_start = time.perf_counter()
        self.batch_start = self.epoch_start

        print(
            f"\nStarting epoch "
            f"{self.current_epoch}/{self.total_epochs}",
            flush=True,
        )

    def on_train_batch_end(self, batch, logs=None):
        now = time.perf_counter()
        completed = batch + 1

        if (
            completed == 1
            or completed % self.interval == 0
            or completed == self.total_batches
        ):
            logs = logs or {}

            print(
                f"Epoch {self.current_epoch}/{self.total_epochs} "
                f"Batch {completed}/{self.total_batches} "
                f"Loss={logs.get('loss', float('nan')):.6f} "
                f"Elapsed={now - self.epoch_start:.1f}s",
                flush=True,
            )

        self.batch_start = now

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        density = logs.get(
            "balanced_density_mse", float("nan")
        )
        tversky = logs.get(
            "soft_tversky_loss", float("nan")
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', float('nan')):.6f}, "
            f"val_loss={logs.get('val_loss', float('nan')):.6f}, "
            f"balanced_mse={density:.6f}, "
            f"soft_tversky={tversky:.6f}, "
            f"weighted_tversky={TVERSKY_WEIGHT * tversky:.6f}",
            flush=True,
        )


# ============================================================
# 4. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 4.1 Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per split for a smoke test.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    # --------------------------------------------------------
    # 4.2 Resolve paths
    # --------------------------------------------------------

    data_root = args.data_root
    output_root = args.output_root

    if not data_root.is_absolute():
        data_root = ROOT / data_root

    if not output_root.is_absolute():
        output_root = ROOT / output_root

    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "logs"
    figures_dir = output_root / "figures"

    # Prevent accidental overwriting of existing experiments.
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output_root}\n"
            "Choose a new --output-root for another run."
        )

    for directory in (
        checkpoints_dir,
        logs_dir,
        figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 4.3 Reproducibility
    # --------------------------------------------------------

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # 4.4 Load E01 preprocessed dataset
    # --------------------------------------------------------

    print("Loading shared E01 migration dataset...", flush=True)

    train_data = make_dataset(
        data_root,
        "train",
        batch_size=args.batch_size,
        limit=args.limit,
        seed=RANDOM_SEED,
    )

    validation_data = make_dataset(
        data_root,
        "validation",
        batch_size=args.batch_size,
        limit=args.limit,
        seed=RANDOM_SEED,
    )

    train_count = len(
        sample_paths(data_root, "train", args.limit)
    )

    validation_count = len(
        sample_paths(data_root, "validation", args.limit)
    )

    total_batches = (
        train_count + args.batch_size - 1
    ) // args.batch_size

    print(f"Experiment: {EXPERIMENT}", flush=True)
    print(f"Training samples: {train_count}", flush=True)
    print(f"Validation samples: {validation_count}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Batches per epoch: {total_batches}", flush=True)

    # --------------------------------------------------------
    # 4.5 Build original E01 3D U-Net
    # --------------------------------------------------------

    model = build_unet3d()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss=e03_combined_loss,
        metrics=[
            balanced_density_mse,
            soft_tversky_loss,
        ],
        jit_compile=False,
    )

    model.summary()

    print("\nE03 loss configuration:", flush=True)
    print("Balanced MSE weight: 1.0", flush=True)
    print(f"Tversky weight: {TVERSKY_WEIGHT}", flush=True)
    print(f"Tversky alpha: {TVERSKY_ALPHA}", flush=True)
    print(f"Tversky beta: {TVERSKY_BETA}", flush=True)

    # --------------------------------------------------------
    # 4.6 Training callbacks
    # --------------------------------------------------------

    checkpoint_path = (
        checkpoints_dir / "best_unet3d.keras"
    )

    history_path = (
        logs_dir / "training_history.csv"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            history_path
        ),

        BatchProgress(
            total_batches=total_batches,
            total_epochs=args.epochs,
        ),
    ]

    # --------------------------------------------------------
    # 4.7 Save experiment configuration
    # --------------------------------------------------------

    config = {
        "experiment": EXPERIMENT,
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
        "checkpoint": str(checkpoint_path),
    }

    config_path = (
        logs_dir / "training_config.json"
    )

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 4.8 Train E03
    # --------------------------------------------------------

    print("\nStarting E03 training...\n", flush=True)

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )

    # --------------------------------------------------------
    # 4.9 Plot training results
    # --------------------------------------------------------

    epoch_numbers = np.arange(
        1,
        len(history.history["loss"]) + 1,
    )

    # Plot the total E03 loss.
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        epoch_numbers,
        history.history["loss"],
        label="Training loss",
    )

    ax.plot(
        epoch_numbers,
        history.history["val_loss"],
        label="Validation loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined loss")
    ax.set_title("E03: Balanced MSE + Soft Tversky")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        figures_dir / "training_loss.png",
        dpi=160,
    )
    plt.close(fig)

    # Plot the individual loss components.
    fig, ax = plt.subplots(figsize=(9, 6))

    if "balanced_density_mse" in history.history:
        ax.plot(
            epoch_numbers,
            history.history["balanced_density_mse"],
            label="Balanced MSE",
        )

    if "soft_tversky_loss" in history.history:
        ax.plot(
            epoch_numbers,
            TVERSKY_WEIGHT * np.asarray(
                history.history["soft_tversky_loss"]
            ),
            label="Weighted Tversky (0.1×)",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss contribution")
    ax.set_title("E03: Training Loss Components")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        figures_dir / "loss_components.png",
        dpi=160,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 4.10 Final training summary
    # --------------------------------------------------------

    best_epoch = (
        int(np.argmin(history.history["val_loss"])) + 1
    )

    best_val_loss = float(
        np.min(history.history["val_loss"])
    )

    print("\n================================", flush=True)
    print("E03 TRAINING COMPLETE", flush=True)
    print("================================", flush=True)

    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best validation loss: {best_val_loss:.6f}", flush=True)
    print(f"Model: {checkpoint_path}", flush=True)
    print(f"History: {history_path}", flush=True)
    print(f"Figures: {figures_dir}", flush=True)


# ============================================================
# 5. RUN
# ============================================================

if __name__ == "__main__":
    main()
