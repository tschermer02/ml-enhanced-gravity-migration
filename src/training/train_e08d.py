"""E08D original training plus configurable E11A-E11E experiments.

Legacy:
  python -m src.training.train_e08d --epochs 100 --batch-size 2

E11:
  python -m src.training.train_e08d --experiment E11A --epochs 100 --batch-size 2
  python -m src.training.train_e08d --experiment E11E --epochs 100 --batch-size 2

E11A/B/D train with exactly the E08D density-only objective.
E11C/E train with the E08D objective plus clean-Gzz physics consistency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.dataset import make_dataset, sample_paths
from src.models.unet3d import build_unet3d
from src.losses.losses import (
    balanced_huber_loss,
    balanced_density_mse,
    soft_tversky_loss,
    vertical_profile_loss,
    make_loss,
    e11_loss_components,
)

ROOT = Path(__file__).resolve().parents[2]
TVERSKY_WEIGHT = 0.1
DEPTH_WEIGHT = 0.02
HUBER_DELTA = 0.1
LEARNING_RATE = 1e-3

EXPERIMENT_DATASETS = {
    "E11BASE": "D07",
    "E11A": "D08",
    "E11B": "D09",
    "E11C": "D07",
    "E11D": "D10",
    "E11E": "D10",
    "E11F": "D10",
}
PHYSICS_EXPERIMENTS = {"E11C", "E11E", "E11F"}

signed_e08d_loss = make_loss(
    weights={"balanced_huber": 1.0, "tversky": TVERSKY_WEIGHT,
             "depth": DEPTH_WEIGHT},
    signed=True, huber_delta=HUBER_DELTA,
)


def signed_balanced_huber(y_true, y_pred):
    return balanced_huber_loss(y_true, y_pred, delta=HUBER_DELTA, signed=True)


def signed_balanced_density_mse(y_true, y_pred):
    return balanced_density_mse(y_true, y_pred, signed=True)


def signed_tversky(y_true, y_pred):
    return soft_tversky_loss(y_true, y_pred, signed=True,
                             subtract_baseline=True)


def signed_vertical_profile(y_true, y_pred):
    return vertical_profile_loss(y_true, y_pred, signed=True)


def build_signed_unet():
    """The same signed tanh head and underlying architecture used by E08D."""
    original = build_unet3d()
    head = tf.keras.layers.Conv3D(
        1, 1, activation="tanh", name="signed_density_prediction"
    )(original.get_layer("decoder1_conv2").output)
    return tf.keras.Model(original.input, head, name="e07_signed_unet3d")


class E11PhysicsTrainer(tf.keras.Model):
    """Train an ordinary signed U-Net while reading clean Gzz and geometry.

    This wrapper is training-only. Checkpoints save `network` as an ordinary
    Keras functional model, so the existing predict.py can load them using
    load_model(..., compile=False).
    """

    def __init__(self, network, physics_weight, physics_normalization_e,
                 tversky_weight=TVERSKY_WEIGHT, depth_weight=DEPTH_WEIGHT):
        super().__init__(name="e11_physics_trainer")
        self.network = network
        self.physics_weight = float(physics_weight)
        self.physics_normalization_e = float(physics_normalization_e)
        self.tversky_weight = float(tversky_weight)
        self.depth_weight = float(depth_weight)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.huber_tracker = tf.keras.metrics.Mean(name="signed_balanced_huber")
        self.tversky_tracker = tf.keras.metrics.Mean(name="signed_tversky")
        self.depth_tracker = tf.keras.metrics.Mean(name="signed_vertical_profile")
        self.physics_tracker = tf.keras.metrics.Mean(name="physics")
        self.density_mse_tracker = tf.keras.metrics.Mean(name="signed_balanced_density_mse")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.huber_tracker,
            self.tversky_tracker,
            self.depth_tracker,
            self.physics_tracker,
            self.density_mse_tracker,
        ]

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

    def _compute(self, x, targets, training):
        pred = self.network(x, training=training)
        parts = e11_loss_components(
            targets["density"],
            pred,
            gravity_clean=targets["gravity_clean"],
            observation_height_m=targets["observation_height_m"],
            dx_m=targets["dx_m"],
            dy_m=targets["dy_m"],
            dz_m=targets["dz_m"],
            physics_weight=self.physics_weight,
            physics_normalization_e=self.physics_normalization_e,
            tversky_weight=self.tversky_weight,
            depth_weight=self.depth_weight,
        )
        return pred, parts

    def _update_metrics(self, truth, pred, parts):
        self.loss_tracker.update_state(parts["total"])
        self.huber_tracker.update_state(parts["balanced_huber"])
        self.tversky_tracker.update_state(parts["tversky"])
        self.depth_tracker.update_state(parts["depth"])
        self.physics_tracker.update_state(parts["physics"])
        self.density_mse_tracker.update_state(
            signed_balanced_density_mse(truth, pred)
        )
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data):
        x, targets = data
        with tf.GradientTape() as tape:
            pred, parts = self._compute(x, targets, training=True)
            total = parts["total"]
        grads = tape.gradient(total, self.network.trainable_variables)
        if any(g is None for g in grads):
            raise RuntimeError("Missing gradient in E11 physics training")
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        return self._update_metrics(targets["density"], pred, parts)

    def test_step(self, data):
        x, targets = data
        pred, parts = self._compute(x, targets, training=False)
        return self._update_metrics(targets["density"], pred, parts)


class SaveBestNetwork(tf.keras.callbacks.Callback):
    """Save only the Keras functional U-Net when validation loss improves."""

    def __init__(self, network, path):
        super().__init__()
        self.network = network
        self.path = str(path)
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_loss")
        if value is not None and np.isfinite(value) and value < self.best:
            self.best = float(value)
            self.network.save(self.path)
            print(f"\nEpoch {epoch + 1}: saved best network to {self.path}")


def plot_history(hist, figures, experiment="E08D", physics=False,
                 tversky_weight=TVERSKY_WEIGHT, depth_weight=DEPTH_WEIGHT,
                 physics_weight=0.0):
    epochs = np.arange(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, hist["loss"], label="Training")
    ax.plot(epochs, hist["val_loss"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Combined loss",
           title=f"{experiment}: training vs validation")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures / "training_loss.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    entries = [
        ("signed_balanced_huber", 1.0),
        ("signed_tversky", tversky_weight),
        ("signed_vertical_profile", depth_weight),
    ]
    if physics:
        entries.append(("physics", physics_weight))  # Plot weighted physics contribution.
    for name, weight in entries:
        for prefix, style in (("", "-"), ("val_", "--")):
            key = prefix + name
            if key in hist:
                ax.plot(epochs, np.asarray(hist[key]) * weight,
                        style, label=key)
    ax.set(xlabel="Epoch", ylabel="Loss component (not all weighted)",
           title=f"{experiment}: loss components")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures / "loss_components.png", dpi=160)
    plt.close(fig)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",
                   choices=("E08D", *EXPERIMENT_DATASETS),
                   default="E08D")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--physics-weight", type=float, default=None)
    p.add_argument("--physics-normalization-e", type=float, default=100.0)
    args = p.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or (
        args.limit is not None and args.limit < 1
    ):
        p.error("epochs, batch-size and limit must be positive")
    if args.physics_normalization_e <= 0:
        p.error("--physics-normalization-e must be positive")

    experiment = args.experiment
    is_e11 = experiment != "E08D"
    expected_physics = experiment in PHYSICS_EXPERIMENTS
    default_weight = 0.01 if expected_physics else 0.0
    physics_weight = (default_weight if args.physics_weight is None
                      else args.physics_weight)
    if physics_weight < 0:
        p.error("--physics-weight cannot be negative")
    if experiment == "E08D" and physics_weight != 0:
        p.error("E08D legacy mode cannot use physics loss")
    if is_e11 and (physics_weight > 0) != expected_physics:
        p.error(f"{experiment} requires physics-weight "
                f"{'> 0' if expected_physics else '0'} to preserve study design")

    default_dataset = ("D06C" if experiment == "E08D"
                       else EXPERIMENT_DATASETS[experiment])
    data_root = resolve(
        args.data_root or f"outputs/ml/training/{default_dataset}/data"
    )
    output = resolve(
        args.output_root or f"outputs/ml/training/{experiment}"
    )
    dirs = [output / "checkpoints", output / "logs", output / "figures"]
    for directory in dirs:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"Existing training outputs in {directory}; use another --output-root"
            )

    train_count = len(sample_paths(data_root, "train", args.limit))
    val_count = len(sample_paths(data_root, "validation", args.limit))

    # E11 always requires the physical grid metadata, even without physics loss.
    if is_e11:
        with (data_root / "config.json").open("r", encoding="utf-8") as stream:
            data_config = json.load(stream)
        if data_config.get("grid") != "e11":
            raise ValueError("E11 requires dataset config grid='e11'")
        if data_config.get("component") != "gzz":
            raise ValueError("E11 requires Gzz observations")
        if not np.isclose(float(data_config.get("dz_m", -1)), 1000.0 / 24.0):
            raise ValueError("E11 vertical grid must span 1 km over 24 layers")
        if not (
            np.isclose(float(data_config.get("dx_m", -1)), 125.0)
            and np.isclose(float(data_config.get("dy_m", -1)), 125.0)
        ):
            raise ValueError("E11 horizontal grid must have 125 m spacing")
    else:
        data_config = None

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    # E11F isolates the relative influence of signed occupancy losses.
    tversky_weight = 0.02 if experiment == "E11F" else TVERSKY_WEIGHT
    depth_weight = 0.005 if experiment == "E11F" else DEPTH_WEIGHT

    tf.keras.utils.set_random_seed(args.seed)
    mode = "physics_clean" if physics_weight > 0 else "density"
    train = make_dataset(
        data_root, "train", batch_size=args.batch_size,
        limit=args.limit, seed=args.seed, targets=mode
    )
    valid = make_dataset(
        data_root, "validation", batch_size=args.batch_size,
        limit=args.limit, seed=args.seed, targets=mode
    )

    network = build_signed_unet()
    checkpoint = dirs[0] / "best_unet3d.keras"

    if physics_weight > 0:
        model = E11PhysicsTrainer(
            network, physics_weight, args.physics_normalization_e,
            tversky_weight=tversky_weight, depth_weight=depth_weight
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            jit_compile=False,
        )
        checkpoint_callback = SaveBestNetwork(network, checkpoint)
    else:
        model = network
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss=signed_e08d_loss,
            metrics=[
                signed_balanced_huber,
                signed_balanced_density_mse,
                signed_tversky,
                signed_vertical_profile,
            ],
            jit_compile=False,
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True, verbose=1
        )

    model.summary()
    callbacks = [
        checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=.5, patience=5,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(dirs[1] / "training_history.csv")),
    ]

    config = dict(
        experiment=experiment,
        comparison_experiment=("E08C" if experiment == "E08D" else "E11BASE"),
        seed=args.seed,
        loss=(f"signed balanced Huber (delta=0.1) + {tversky_weight} signed Tversky "
              f"+ {depth_weight} depth profile"
              + (f" + {physics_weight} clean-Gzz physics" if physics_weight else "")),
        huber_delta_gcm3=HUBER_DELTA,
        output="tanh [-1,1] g/cm3",
        epochs=args.epochs,
        batch_size=args.batch_size,
        training_data=str(data_root),
        learning_rate=LEARNING_RATE,
        checkpoint=str(checkpoint),
        train_count=train_count,
        validation_count=val_count,
    )
    if is_e11:
        config.update(
            dataset=default_dataset,
            physics_weight=physics_weight,
            tversky_weight=tversky_weight,
            depth_weight=depth_weight,
            physics_normalization_e=args.physics_normalization_e,
            physics_target="gravity_clean" if physics_weight else "none",
            grid=data_config["grid"],
            dx_m=data_config["dx_m"],
            dy_m=data_config["dy_m"],
            dz_m=data_config["dz_m"],
            observation_heights_m=data_config.get("observation_heights_m"),
        )

    (dirs[1] / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    history = model.fit(
        train, validation_data=valid, epochs=args.epochs,
        callbacks=callbacks, verbose=2
    ).history
    plot_history(history, dirs[2], experiment=experiment,
                 physics=(physics_weight > 0),
                 tversky_weight=tversky_weight, depth_weight=depth_weight,
                 physics_weight=physics_weight)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    print(f"{experiment} training complete; best epoch {best_epoch}; "
          f"checkpoint {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
