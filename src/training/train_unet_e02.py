
"""
Train experiment E02 using balanced Huber loss.

E01: Balanced density MSE
E02: Balanced Huber loss

The architecture, dataset, preprocessing, and training
hyperparameters remain unchanged.

No gravity physics loss or structural loss is used.
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


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gravity_migration_ml.ml.dataset import (
    make_dataset,
    sample_paths
)

from gravity_migration_ml.ml.losses import balanced_huber_loss
from gravity_migration_ml.ml.unet3d import build_unet3d


# ============================================================
# 2. PARAMETERS
# ============================================================

EXPERIMENT = "E02_balanced_huber"

LEARNING_RATE = 1e-3
BATCH_SIZE = 2
MAX_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5

RANDOM_SEED = 42
HUBER_DELTA = 0.1

# Use the same preprocessed migration data as E01
DATA_ROOT = (
    ROOT / "outputs" / "ml" / "training" / "E01" / "data"
)

# Save to a separate folder to preserve the previous E02
OUTPUT_ROOT = (
    ROOT / "outputs" / "ml" / "training" / EXPERIMENT
)


# ============================================================
# 3. BATCH PROGRESS CALLBACK
# ============================================================

class BatchProgress(tf.keras.callbacks.Callback):
    """Print training progress during each epoch."""

    def __init__(self, total_batches, total_epochs, interval=10):
        super().__init__()

        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.interval = interval

    def on_epoch_begin(self, epoch, logs=None):

        self.current_epoch = epoch + 1
        self.epoch_start = time.perf_counter()
        self.batch_start = self.epoch_start

        self.first_batch_time = None
        self.later_batch_times = []

        print(
            f"\nStarting epoch "
            f"{self.current_epoch}/{self.total_epochs}",
            flush=True
        )

    def on_train_batch_end(self, batch, logs=None):

        now = time.perf_counter()
        batch_time = now - self.batch_start

        if batch == 0:
            self.first_batch_time = batch_time
        else:
            self.later_batch_times.append(batch_time)

        completed = batch + 1

        if (
            completed == 1
            or completed % self.interval == 0
            or completed == self.total_batches
        ):

            logs = logs or {}

            elapsed = now - self.epoch_start

            later_average = (
                float(np.mean(self.later_batch_times))
                if self.later_batch_times
                else float("nan")
            )

            print(
                f"Epoch {self.current_epoch}/{self.total_epochs} "
                f"Batch {completed}/{self.total_batches} "
                f"Loss={logs.get('loss', float('nan')):.6f} "
                f"Batch time={batch_time:.2f}s "
                f"Elapsed={elapsed:.1f}s",
                flush=True
            )

            if completed == self.total_batches:
                print(
                    f"First batch: "
                    f"{self.first_batch_time:.2f}s "
                    f"Later batch average: "
                    f"{later_average:.2f}s",
                    flush=True
                )

        self.batch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        print(
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', float('nan')):.6f}, "
            f"val_loss={logs.get('val_loss', float('nan')):.6f}",
            flush=True
        )


# ============================================================
# 4. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 4.1 Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per split for testing."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT
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

    # Protect old results from accidental overwriting
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output_root}\n"
            "Use --output-root with a new directory."
        )

    for directory in (
        checkpoints_dir,
        logs_dir,
        figures_dir
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # 4.3 Set random seeds
    # --------------------------------------------------------

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # 4.4 Load preprocessed migration datasets
    # --------------------------------------------------------

    print("Loading E01 preprocessed migration datasets...")

    train_data = make_dataset(
        data_root,
        "train",
        batch_size=args.batch_size,
        limit=args.limit,
        seed=RANDOM_SEED
    )

    validation_data = make_dataset(
        data_root,
        "validation",
        batch_size=args.batch_size,
        limit=args.limit,
        seed=RANDOM_SEED
    )

    train_count = len(
        sample_paths(
            data_root,
            "train",
            args.limit
        )
    )

    validation_count = len(
        sample_paths(
            data_root,
            "validation",
            args.limit
        )
    )

    total_batches = (
        train_count + args.batch_size - 1
    ) // args.batch_size

    print(f"Experiment: {EXPERIMENT}")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {validation_count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per epoch: {total_batches}")

    # --------------------------------------------------------
    # 4.5 Build original E01 3D U-Net
    # --------------------------------------------------------

    model = build_unet3d()

    # Only experimental change: balanced Huber loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=balanced_huber_loss,
        jit_compile=False
    )

    model.summary()

    # --------------------------------------------------------
    # 4.6 Define callbacks
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
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            history_path
        ),

        BatchProgress(
            total_batches=total_batches,
            total_epochs=args.epochs
        )

    ]

    # --------------------------------------------------------
    # 4.7 Save configuration
    # --------------------------------------------------------

    config = {
        "experiment": EXPERIMENT,
        "architecture": "E01 3D U-Net",
        "loss": "balanced_huber",
        "huber_delta_gcm3": HUBER_DELTA,
        "body_weight": 0.5,
        "background_weight": 0.5,
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
        "checkpoint": str(checkpoint_path)
    }

    config_path = (
        logs_dir / "training_config.json"
    )

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 4.8 Train network
    # --------------------------------------------------------

    print("\nStarting E02 training...\n")

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0
    )

    # --------------------------------------------------------
    # 4.9 Plot training results
    # --------------------------------------------------------

    epoch_numbers = np.arange(
        1,
        len(history.history["loss"]) + 1
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        epoch_numbers,
        history.history["loss"],
        label="Training loss"
    )

    ax.plot(
        epoch_numbers,
        history.history["val_loss"],
        label="Validation loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced Huber loss")
    ax.set_title("E02: Balanced Huber Training")

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    figure_path = (
        figures_dir / "training_loss.png"
    )

    fig.savefig(
        figure_path,
        dpi=160
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 4.10 Final summary
    # --------------------------------------------------------

    best_epoch = (
        int(np.argmin(history.history["val_loss"])) + 1
    )

    best_val_loss = float(
        np.min(history.history["val_loss"])
    )

    print("\n================================")
    print("E02 TRAINING COMPLETE")
    print("================================")

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Figure: {figure_path}")


# ============================================================
# 5. RUN
# ============================================================

if __name__ == "__main__":
    main()
