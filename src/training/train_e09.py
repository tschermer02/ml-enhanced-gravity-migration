"""E09: E08D balanced-Huber 3D U-Net with differentiable Gzz data consistency.

The data term uses the observed Gzz saved in D06C NPZ samples and the same
FFT convolution kernels/crop as src.physics.gravity_forward.calculate_gzz.

Run from repository root:
 python -m src.training.train_e09 --data-root outputs/ml/training/D06C/data \
     --output-root outputs/ml/training/E09 --epochs 100 --batch-size 2
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.fft import rfft2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import make_dataset, sample_paths
from src.models.unet3d import build_unet3d
from src.losses.losses import (balanced_huber_loss, balanced_density_mse,
                              soft_tversky_loss, vertical_profile_loss)
from src.physics.gravity_forward import gzz_kernel, CELL_SIZE, G

ROOT = Path(__file__).resolve().parents[2]
HUBER_DELTA = 0.1
TVERSKY_WEIGHT = 0.1
DEPTH_WEIGHT = 0.02
PHYSICS_WEIGHT = 0.01  # First experimental value; inspect component losses
LEARNING_RATE = 1e-3
N_Z, N_Y, N_X = 24, 64, 64
FFT_SIZE = 207  # Full convolution of 64 x 64 layer with 144 x 144 kernel
CROP_START = 63
CROP_END = 144  # output is 81 x 81


def _frequency_kernels():
    """Constant spectra, consistent with scipy.signal.fftconvolve(..., mode='full')."""
    kernels = [rfft2(gzz_kernel((z + 0.5) * CELL_SIZE), s=(FFT_SIZE, FFT_SIZE))
               for z in range(N_Z)]
    return tf.constant(np.stack(kernels).astype(np.complex64))


class GzzForward(tf.keras.layers.Layer):
    """Differentiable batch Gzz forward, returns (batch, 81, 81, 1) Eotvos."""
    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.kernels = _frequency_kernels()
        self.multiplier = tf.constant(G * CELL_SIZE ** 3 * 1e12, tf.float32)

    def call(self, density):
        # density: B,Z,Y,X,1; FFT on lateral axes only.
        rho = tf.cast(tf.squeeze(density, axis=-1), tf.float32)
        padded = tf.pad(rho, [[0, 0], [0, 0], [0, FFT_SIZE - N_Y],
                              [0, FFT_SIZE - N_X]])
        transformed = tf.signal.rfft2d(padded)  # B,Z,207,104
        full_spectrum = tf.reduce_sum(transformed * self.kernels[None, ...], axis=1)
        full = tf.signal.irfft2d(full_spectrum, fft_length=[FFT_SIZE, FFT_SIZE])
        field = full[:, CROP_START:CROP_END, CROP_START:CROP_END]
        return (field * self.multiplier)[..., None]


def _loss_terms(y_true, y_pred):
    huber = balanced_huber_loss(y_true, y_pred, delta=HUBER_DELTA, signed=True)
    tversky = soft_tversky_loss(y_true, y_pred, signed=True,
                               subtract_baseline=True)
    depth = vertical_profile_loss(y_true, y_pred, signed=True)
    return huber, tversky, depth


class E09Model(tf.keras.Model):
    def __init__(self, density_net, physics_weight=PHYSICS_WEIGHT):
        super().__init__(name="e09_physics_unet3d")
        self.density_net = density_net
        self.gzz_forward = GzzForward(name="differentiable_gzz")
        self.physics_weight = float(physics_weight)
        self.metrics_loss = {
            k: tf.keras.metrics.Mean(name=k) for k in
            ("loss", "balanced_huber", "signed_balanced_density_mse",
             "signed_tversky", "signed_vertical_profile", "physics_loss")
        }

    @property
    def metrics(self):
        return list(self.metrics_loss.values())

    def call(self, inputs, training=False):
        return self.density_net(inputs, training=training)

    @staticmethod
    def _relative_physics_loss(observed, predicted):
        # Normalization uses only observed data; stable, sign-invariant per sample.
        obs = tf.cast(observed, tf.float32)
        scale = tf.reduce_max(tf.abs(obs), axis=[1, 2, 3], keepdims=True)
        scale = tf.maximum(scale, tf.constant(1e-4, tf.float32))
        return tf.reduce_mean(tf.square((predicted - obs) / scale))

    def compute_e09_losses(self, x, targets, training):
        density_true = targets["density"]
        observed = targets["gravity"]
        prediction = self.density_net(x, training=training)
        huber, tversky, depth = _loss_terms(density_true, prediction)
        gzz_prediction = self.gzz_forward(prediction)
        physics = self._relative_physics_loss(observed, gzz_prediction)
        total = huber + TVERSKY_WEIGHT*tversky + DEPTH_WEIGHT*depth + self.physics_weight*physics
        diagnostic_mse = balanced_density_mse(density_true, prediction, signed=True)
        terms = dict(loss=total, balanced_huber=huber,
                     signed_balanced_density_mse=diagnostic_mse,
                     signed_tversky=tversky, signed_vertical_profile=depth,
                     physics_loss=physics)
        return terms

    def train_step(self, data):
        x, targets = data
        with tf.GradientTape() as tape:
            terms = self.compute_e09_losses(x, targets, training=True)
        grads = tape.gradient(terms["loss"], self.density_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.density_net.trainable_variables))
        for name, tracker in self.metrics_loss.items():
            tracker.update_state(terms[name])
        return {name: tracker.result() for name, tracker in self.metrics_loss.items()}

    def test_step(self, data):
        x, targets = data
        terms = self.compute_e09_losses(x, targets, training=False)
        for name, tracker in self.metrics_loss.items():
            tracker.update_state(terms[name])
        return {name: tracker.result() for name, tracker in self.metrics_loss.items()}


def plot_history(history, directory, physics_weight):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, history["loss"], label="Training")
    ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Combined loss", title="E09: Gzz-constrained training")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(directory / "training_loss.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, weight in (("balanced_huber", 1.0), ("signed_tversky", TVERSKY_WEIGHT),
                        ("signed_vertical_profile", DEPTH_WEIGHT),
                        ("physics_loss", physics_weight)):
        for prefix, style in (("", "-"), ("val_", "--")):
            name = prefix + key
            if name in history:
                ax.plot(epochs, np.asarray(history[name])*weight, style, label=name)
    ax.set(xlabel="Epoch", ylabel="Weighted loss", title="E09: loss components")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(directory / "loss_components.png", dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int)
    p.add_argument("--physics-weight", type=float, default=PHYSICS_WEIGHT)
    p.add_argument("--data-root", default="outputs/ml/training/D06C/data")
    p.add_argument("--output-root", default="outputs/ml/training/E09")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.physics_weight < 0 or (args.limit is not None and args.limit < 1):
        p.error("Invalid training arguments")
    resolve = lambda v: Path(v) if Path(v).is_absolute() else ROOT / v
    data_root, output = resolve(args.data_root), resolve(args.output_root)
    train_count = len(sample_paths(data_root, "train", args.limit))
    val_count = len(sample_paths(data_root, "validation", args.limit))
    # Verify that the physics data are truly Gzz, not Gz.
    for split in ("train", "validation"):
        with np.load(sample_paths(data_root, split, 1)[0]) as sample:
            component = str(sample["observation_component"].item()).lower()
            if component != "gzz" or "gravity" not in sample:
                raise ValueError(f"{split} must contain observed Gzz in gravity array")
    dirs = [output / name for name in ("checkpoints", "logs", "figures")]
    for directory in dirs:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Existing output directory is not empty: {directory}")
    for directory in dirs: directory.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    train = make_dataset(data_root, "train", batch_size=args.batch_size, limit=args.limit,
                         seed=args.seed, targets="physics")
    valid = make_dataset(data_root, "validation", batch_size=args.batch_size, limit=args.limit,
                         seed=args.seed, targets="physics")
    base = build_unet3d()
    head = tf.keras.layers.Conv3D(1, 1, activation="tanh", name="signed_density_prediction")(
        base.get_layer("decoder1_conv2").output)
    density_net = tf.keras.Model(base.input, head, name="e09_signed_density_unet")
    model = E09Model(density_net, physics_weight=args.physics_weight)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), jit_compile=False)
    # Save the density network alone, preserving compatibility with prediction/evaluation.
    checkpoint = dirs[0] / "best_unet3d.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                        restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5,
                                             patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(dirs[1] / "training_history.csv")),
    ]
    # Custom callback saves only density_net; otherwise the subclassed training
    # wrapper (which requires observed gravity) would not load via compile=False.
    class SaveDensityNet(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = np.inf
        def on_epoch_end(self, epoch, logs=None):
            current = float((logs or {}).get("val_loss", np.inf))
            if current < self.best:
                self.best = current
                density_net.save(str(checkpoint))
                print(f"Epoch {epoch+1}: saved E09 density checkpoint (val_loss={current:.6f})", flush=True)
    callbacks[0] = SaveDensityNet()
    config = dict(experiment="E09", baseline="E08D", dataset=str(data_root),
                  objective="balanced Huber + 0.1 signed Tversky + 0.02 depth + weighted relative Gzz MSE",
                  huber_delta=HUBER_DELTA, physics_weight=args.physics_weight,
                  physics_normalization="per-sample max absolute observed Gzz, floor=1e-4 E",
                  geometry_shape=[N_Z,N_Y,N_X], observed_grid=[81,81],
                  learning_rate=LEARNING_RATE, seed=args.seed, epochs=args.epochs,
                  batch_size=args.batch_size, train_count=train_count,
                  validation_count=val_count, checkpoint=str(checkpoint))
    (dirs[1]/"training_config.json").write_text(json.dumps(config, indent=2))
    hist = model.fit(train, validation_data=valid, epochs=args.epochs,
                     callbacks=callbacks, verbose=2).history
    plot_history(hist, dirs[2], args.physics_weight)
    best_epoch = int(np.argmin(hist["val_loss"]))+1
    print(f"E09 complete: best epoch={best_epoch}, checkpoint={checkpoint}", flush=True)

if __name__ == "__main__":
    main()
