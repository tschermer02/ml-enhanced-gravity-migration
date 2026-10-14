"""Train E02 with Huber, gravity-consistency, and structural losses."""
from __future__ import annotations

import argparse
import csv
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

from gravity_migration_ml.ml.e02_dataset import make_e02_dataset  # noqa: E402
from gravity_migration_ml.ml.e02_trainer import E02Trainer  # noqa: E402
from gravity_migration_ml.ml.unet3d import build_unet3d  # noqa: E402
from gravity_migration_ml.ml.dataset import sample_paths  # noqa: E402

# Controlled-comparison settings copied from E01.
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
RANDOM_SEED = 42

# E02 objective weights; adjust here or with the command line.
LAMBDA_MODEL = 1.0
LAMBDA_PHYS = 0.1
LAMBDA_STRUCT = 0.1
LAMBDA_REG = 0.0
HUBER_DELTA = 0.1
# Initial-network checks produced an unnormalized physics loss near 5.7 versus
# a Huber loss near 0.045. A 3.5 mGal scale makes lambda_phys=0.1 comparable at
# initialization. This affects only the loss residual, never physical outputs.
GRAVITY_NORMALIZATION_MGAL = 3.5
CELL_SPACING_M = 10.0

E01_DATA_ROOT = ROOT / "outputs" / "ml" / "training" / "E01" / "data"
GRAVITY_ROOT = SRC / "data" / "dataset" / "forward_models"
OUTPUT_ROOT = ROOT / "outputs" / "ml" / "training" / "E02"


class E02EpochPrinter(tf.keras.callbacks.Callback):
    """Print each individual train/validation loss at epoch end."""

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        print(
            f"Epoch {epoch + 1}: total={logs.get('total_loss', float('nan')):.6g}, "
            f"model={logs.get('model_loss', float('nan')):.6g}, "
            f"physics={logs.get('physics_loss', float('nan')):.6g}, "
            f"struct={logs.get('struct_loss', float('nan')):.6g}, "
            f"val_total={logs.get('val_total_loss', float('nan')):.6g}, "
            f"val_model={logs.get('val_model_loss', float('nan')):.6g}, "
            f"val_physics={logs.get('val_physics_loss', float('nan')):.6g}, "
            f"val_struct={logs.get('val_struct_loss', float('nan')):.6g}",
            flush=True,
        )


class E02BatchProgress(tf.keras.callbacks.Callback):
    """Print sparse batch progress and compare first batch with later batches."""

    def __init__(self, total_batches: int, total_epochs: int, interval: int = 10):
        super().__init__()
        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.interval = interval

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        self.current_epoch = epoch + 1
        self.epoch_started = time.perf_counter()
        self.batch_started = self.epoch_started
        self.first_batch_seconds: float | None = None
        self.later_batch_seconds: list[float] = []
        print(f"Starting epoch {epoch + 1}/{self.total_epochs}", flush=True)

    def on_train_batch_end(self, batch: int, logs: dict[str, float] | None = None) -> None:
        now = time.perf_counter()
        batch_seconds = now - self.batch_started
        if batch == 0:
            self.first_batch_seconds = batch_seconds
        else:
            self.later_batch_seconds.append(batch_seconds)
        completed = batch + 1
        if completed == 1 or completed % self.interval == 0 or completed == self.total_batches:
            logs = logs or {}
            later_average = (
                float(np.mean(self.later_batch_seconds))
                if self.later_batch_seconds else float("nan")
            )
            first_text = (
                f"{self.first_batch_seconds:.2f}s"
                if self.first_batch_seconds is not None else "n/a"
            )
            later_text = f"{later_average:.2f}s" if np.isfinite(later_average) else "n/a"
            print(
                f"Epoch {self.current_epoch}/{self.total_epochs} "
                f"batch {completed}/{self.total_batches}: "
                f"total_loss={logs.get('total_loss', float('nan')):.5g}; "
                f"batch={batch_seconds:.2f}s; first={first_text}; "
                f"later_avg={later_text}; elapsed={now - self.epoch_started:.1f}s",
                flush=True,
            )
        self.batch_started = now


class SaveBestNetwork(tf.keras.callbacks.Callback):
    """Save only the base U-Net, keeping E02 training wrapper out of inference."""

    def __init__(self, output_path: Path):
        super().__init__()
        self.output_path = output_path
        self.best = float("inf")

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        value = float((logs or {}).get("val_total_loss", float("inf")))
        if np.isfinite(value) and value < self.best:
            self.best = value
            self.model.network.save(self.output_path)
            print(f"Saved best E02 network: {self.output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=E01_DATA_ROOT,
                        help="Use the unchanged, preprocessed E01 migration dataset.")
    parser.add_argument("--gravity-root", type=Path, default=GRAVITY_ROOT,
                        help="Forward-model splits containing the observed gravity maps.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit each split for a short smoke test.")
    parser.add_argument("--lambda-model", type=float, default=LAMBDA_MODEL)
    parser.add_argument("--lambda-phys", type=float, default=LAMBDA_PHYS)
    parser.add_argument("--lambda-struct", type=float, default=LAMBDA_STRUCT)
    parser.add_argument("--lambda-reg", type=float, default=LAMBDA_REG)
    parser.add_argument("--huber-delta", type=float, default=HUBER_DELTA)
    parser.add_argument("--gravity-normalization-mgal", type=float,
                        default=GRAVITY_NORMALIZATION_MGAL,
                        help="Divide gravity residuals in the physics loss by this mGal scale.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.huber_delta <= 0 or args.gravity_normalization_mgal <= 0:
        parser.error("Huber delta and gravity normalization must be positive")

    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    gravity_root = args.gravity_root if args.gravity_root.is_absolute() else ROOT / args.gravity_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    checkpoints = output_root / "checkpoints"
    logs = output_root / "logs"
    figures = output_root / "figures"
    for directory in (checkpoints, logs, figures):
        directory.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    train_data = make_e02_dataset(
        data_root, gravity_root, "train", args.batch_size,
        limit=args.limit, seed=RANDOM_SEED,
    )
    validation_data = make_e02_dataset(
        data_root, gravity_root, "validation", args.batch_size,
        limit=args.limit, seed=RANDOM_SEED,
    )
    train_sample_count = len(sample_paths(data_root, "train", args.limit))
    total_batches = (train_sample_count + args.batch_size - 1) // args.batch_size

    weights = {
        "lambda_model": args.lambda_model,
        "lambda_phys": args.lambda_phys,
        "lambda_struct": args.lambda_struct,
        "lambda_reg": args.lambda_reg,
    }
    trainer = E02Trainer(
        build_unet3d(), weights,
        huber_delta=args.huber_delta,
        gravity_normalization_mgal=args.gravity_normalization_mgal,
        cell_spacing_m=CELL_SPACING_M,
    )
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        jit_compile=False,
    )

    checkpoint_path = checkpoints / "best_unet3d.keras"
    callbacks = [
        SaveBestNetwork(checkpoint_path),
        E02BatchProgress(total_batches, args.epochs),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_total_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_total_loss", factor=0.5, patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6, verbose=1, mode="min",
        ),
        E02EpochPrinter(),
    ]
    config = {
        "experiment": "E02",
        "architecture": "E01 build_unet3d unchanged",
        "data_root": str(data_root),
        "observed_gravity_root": str(gravity_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "seed": RANDOM_SEED,
        "loss_weights": weights,
        "huber_delta_gcm3": args.huber_delta,
        "gravity_normalization_mgal": args.gravity_normalization_mgal,
        "gravity_normalization_note": (
            "Only residuals in the physics loss are divided by this value; "
            "the forward operator output remains physical mGal."
        ),
        "structural_cell_spacing_m": CELL_SPACING_M,
        "regularization": "no penalty; lambda_reg is zero placeholder",
        "limit_per_split": args.limit,
        "checkpoint": str(checkpoint_path),
    }
    (logs / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = trainer.fit(
        train_data, validation_data=validation_data, epochs=args.epochs,
        callbacks=callbacks, verbose=0,
    )
    history_path = logs / "training_history.csv"
    columns = (
        "total_loss", "model_loss", "physics_loss", "struct_loss",
        "val_total_loss", "val_model_loss", "val_physics_loss", "val_struct_loss",
    )
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", *columns))
        writer.writeheader()
        for epoch, values in enumerate(zip(*(history.history[key] for key in columns)), 1):
            writer.writerow({"epoch": epoch, **dict(zip(columns, values))})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, key, val_key, label in (
        (axes[0, 0], "total_loss", "val_total_loss", "Total loss"),
        (axes[0, 1], "model_loss", "val_model_loss", "Huber model loss"),
        (axes[1, 0], "physics_loss", "val_physics_loss", "Gravity physics loss"),
        (axes[1, 1], "struct_loss", "val_struct_loss", "Structural gradient loss"),
    ):
        axis.plot(history.history[key], label="Train")
        axis.plot(history.history[val_key], label="Validation")
        axis.set(title=label, xlabel="Epoch", ylabel="Loss")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(figures / "training_losses.png", dpi=160)
    plt.close(fig)
    print(f"Best E02 model: {checkpoint_path}")


if __name__ == "__main__":
    main()
