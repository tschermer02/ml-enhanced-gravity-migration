"""E06B: signed mixed-geology inversion with balanced Huber instead of MSE.

Controlled comparison to E06A:
  E06A = signed balanced MSE + 0.1 signed Tversky + 0.02 depth
  E06B = signed balanced Huber (delta=0.1 g/cm3) + SAME other terms

Both runs must use the SAME D04B dataset for a valid comparison.

Example:
 python -m src.training.train_e06b --data-root outputs/ml/training/D04B/data \
     --output-root outputs/ml/training/E06B --epochs 100 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.dataset import make_dataset, sample_paths
from src.models.unet3d import build_unet3d
from src.losses.losses import (
    balanced_huber_loss, balanced_density_mse, soft_tversky_loss,
    vertical_profile_loss, make_loss,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "E06B"
SEED = 42
LEARNING_RATE = 1e-3
HUBER_DELTA = 0.1
TVERSKY_WEIGHT = 0.1
DEPTH_WEIGHT = 0.02
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5

# Identical to E06A except "balanced_mse" -> "balanced_huber".
signed_e06b_loss = make_loss(
    weights={"balanced_huber": 1.0, "tversky": TVERSKY_WEIGHT, "depth": DEPTH_WEIGHT},
    signed=True,
    huber_delta=HUBER_DELTA,
)


def signed_balanced_huber(y_true, y_pred):
    return balanced_huber_loss(y_true, y_pred, delta=HUBER_DELTA, signed=True)


def signed_balanced_mse(y_true, y_pred):
    """Diagnostic only: compare density MSE between E06A and E06B."""
    return balanced_density_mse(y_true, y_pred, signed=True)


def signed_tversky(y_true, y_pred):
    return soft_tversky_loss(y_true, y_pred, signed=True, subtract_baseline=True)


def signed_vertical_profile(y_true, y_pred):
    return vertical_profile_loss(y_true, y_pred, signed=True)


class BatchProgress(tf.keras.callbacks.Callback):
    def __init__(self, batches, total_epochs, interval=10):
        super().__init__()
        self.batches = batches
        self.total_epochs = total_epochs
        self.interval = interval

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.started = time.perf_counter()
        print(f"\nStarting epoch {self.epoch}/{self.total_epochs}", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        count = batch + 1
        if count == 1 or count % self.interval == 0 or count == self.batches:
            loss = (logs or {}).get("loss", float("nan"))
            print(f"Epoch {self.epoch} Batch {count}/{self.batches}: "
                  f"loss={loss:.6f}, elapsed={time.perf_counter()-self.started:.1f}s", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        def metric(name):
            return float(logs.get(name, float("nan")))
        print(f"Epoch {epoch+1}: loss={metric('loss'):.6f}, val_loss={metric('val_loss'):.6f}, "
              f"Huber={metric('signed_balanced_huber'):.6f}, "
              f"val_Huber={metric('val_signed_balanced_huber'):.6f}, "
              f"weighted_Tversky={TVERSKY_WEIGHT * metric('signed_tversky'):.6f}, "
              f"val_weighted_Tversky={TVERSKY_WEIGHT * metric('val_signed_tversky'):.6f}, "
              f"weighted_depth={DEPTH_WEIGHT * metric('signed_vertical_profile'):.6f}, "
              f"val_weighted_depth={DEPTH_WEIGHT * metric('val_signed_vertical_profile'):.6f}", flush=True)


def plot_history(history, outdir):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, history["loss"], label="Training")
    ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Combined loss", title="E06B: Balanced Huber + Tversky + Depth")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "training_loss.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, weight, label in (
        ("signed_balanced_huber", 1., "Balanced Huber"),
        ("signed_tversky", TVERSKY_WEIGHT, "Weighted Tversky"),
        ("signed_vertical_profile", DEPTH_WEIGHT, "Weighted depth profile"),
    ):
        if name in history:
            ax.plot(epochs, weight * np.asarray(history[name]), label=label)
        if "val_" + name in history:
            ax.plot(epochs, weight * np.asarray(history["val_" + name]), linestyle="--", label="Validation " + label)
    ax.set(xlabel="Epoch", ylabel="Weighted loss contribution", title="E06B: Loss components")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_components.png", dpi=160)
    plt.close(fig)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("outputs/ml/training/D04B/data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ml/training/E06B"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("epochs, batch size, and limit must be positive")

    data_root, output_root = resolve(args.data_root), resolve(args.output_root)
    checkpoints, logs, figures = (output_root / d for d in ("checkpoints", "logs", "figures"))
    for directory in (checkpoints, logs, figures):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing E06B results: {directory}")
    train_count = len(sample_paths(data_root, "train", args.limit))
    val_count = len(sample_paths(data_root, "validation", args.limit))
    for directory in (checkpoints, logs, figures):
        directory.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    train_data = make_dataset(data_root, "train", batch_size=args.batch_size, limit=args.limit, seed=SEED)
    validation_data = make_dataset(data_root, "validation", batch_size=args.batch_size, limit=args.limit, seed=SEED)

    original = build_unet3d()
    head = tf.keras.layers.Conv3D(1, 1, activation="tanh", name="signed_density_prediction")(
        original.get_layer("decoder1_conv2").output
    )
    model = tf.keras.Model(original.input, head, name="e06_signed_unet3d")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=signed_e06b_loss,
        metrics=[signed_balanced_huber, signed_balanced_mse, signed_tversky, signed_vertical_profile],
        jit_compile=False,
    )
    checkpoint = checkpoints / "best_unet3d.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=REDUCE_LR_PATIENCE,
                                             min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(logs / "training_history.csv")),
        BatchProgress((train_count + args.batch_size - 1) // args.batch_size, args.epochs),
    ]
    config = {
        "experiment": EXPERIMENT, "comparison_experiment": "E06A",
        "changed_variable": "signed balanced MSE replaced by signed balanced Huber",
        "loss": "1.0 balanced_huber + 0.1 signed_tversky + 0.02 signed_vertical_profile",
        "huber_delta_gcm3": HUBER_DELTA, "tversky_weight": TVERSKY_WEIGHT,
        "depth_weight": DEPTH_WEIGHT, "signed": True,
        "architecture": "E06A 3D U-Net with tanh output", "seed": SEED,
        "learning_rate": LEARNING_RATE, "batch_size": args.batch_size,
        "max_epochs": args.epochs, "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE, "train_count": train_count,
        "validation_count": val_count, "data_root": str(data_root),
        "checkpoint": str(checkpoint),
    }
    (logs / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"E06B: {train_count} train / {val_count} validation samples, dataset={data_root}", flush=True)
    hist = model.fit(train_data, validation_data=validation_data, epochs=args.epochs,
                     callbacks=callbacks, verbose=0).history
    plot_history(hist, figures)
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    print(f"E06B complete. Best epoch={best_epoch}, "
          f"best val loss={np.min(hist['val_loss']):.6f}, checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
