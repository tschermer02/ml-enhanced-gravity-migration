"""Numerical and gradient checks for E02's TensorFlow gravity operator."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from physics.gravity_forward import calculate_gravity
from gravity_migration_ml.ml.e02_dataset import load_e02_sample
from gravity_migration_ml.ml.e02_losses import e02_loss_components
from gravity_migration_ml.ml.e02_physics import gravity_forward_tf


VOLUME_SHAPE = (24, 64, 64)


def _as_tensor(density: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(density[None, ..., None], dtype=tf.float32)


def test_zero_density_returns_zero_gravity() -> None:
    result = gravity_forward_tf(_as_tensor(np.zeros(VOLUME_SHAPE, np.float32)))
    assert result.shape == (1, 81, 81, 1)
    np.testing.assert_allclose(result.numpy(), 0.0, atol=1e-8)


def test_prism_matches_numpy_forward_model() -> None:
    density = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    density[3:7, 25:35, 29:41] = 0.65
    predicted = gravity_forward_tf(_as_tensor(density)).numpy()[0, ..., 0]
    reference = calculate_gravity(density)
    np.testing.assert_allclose(predicted, reference, rtol=2e-4, atol=2e-7)


def test_sparse_random_density_matches_numpy_forward_model() -> None:
    rng = np.random.default_rng(2026)
    density = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    occupied = rng.random(VOLUME_SHAPE) < 0.02
    density[occupied] = rng.uniform(0.05, 0.95, size=int(occupied.sum()))
    predicted = gravity_forward_tf(_as_tensor(density)).numpy()[0, ..., 0]
    reference = calculate_gravity(density)
    assert predicted.shape == (81, 81)
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(predicted, reference, rtol=3e-4, atol=3e-7)


def test_gradient_is_finite_with_respect_to_density() -> None:
    density = tf.Variable(tf.ones((1, *VOLUME_SHAPE, 1), dtype=tf.float32) * 0.1)
    with tf.GradientTape() as tape:
        gravity = gravity_forward_tf(density)
        objective = tf.reduce_mean(tf.square(gravity))
    gradient = tape.gradient(objective, density)
    assert gradient is not None
    assert gradient.shape == density.shape
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())


def test_gradient_descent_reduces_gravity_misfit() -> None:
    target_density = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    target_density[4:7, 30:36, 28:35] = 0.4
    target_gravity = tf.constant(calculate_gravity(target_density)[None, ..., None], tf.float32)
    estimate = tf.Variable(tf.zeros((1, *VOLUME_SHAPE, 1), tf.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

    def misfit() -> tf.Tensor:
        return tf.reduce_mean(tf.square(gravity_forward_tf(estimate) - target_gravity))

    initial = float(misfit().numpy())
    for _ in range(5):
        with tf.GradientTape() as tape:
            loss = misfit()
        optimizer.apply_gradients([(tape.gradient(loss, estimate), estimate)])
    assert float(misfit().numpy()) < initial


def test_e02_loader_and_combined_loss_are_finite(tmp_path) -> None:
    from gravity_migration_ml.ml.e02_dataset import load_e02_sample

    data_root = tmp_path / "migration_data" / "train"
    gravity_root = tmp_path / "forward_models"
    sample_path = data_root / "sample_000000.npz"
    gravity_path = gravity_root / "train" / sample_path.name
    data_root.mkdir(parents=True)
    gravity_path.parent.mkdir(parents=True)
    density_data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    density_data[3:6, 20:26, 28:34] = 0.4
    gravity_data = calculate_gravity(density_data).astype(np.float32)
    np.savez_compressed(
        sample_path,
        density=density_data,
        migration=np.zeros(VOLUME_SHAPE, dtype=np.float32),
        migration_input=np.zeros(VOLUME_SHAPE, dtype=np.float32),
        migration_scale=np.float64(0.0),
    )
    np.savez_compressed(gravity_path, gravity=gravity_data)
    migration_input, targets = load_e02_sample(sample_path, gravity_root)
    assert migration_input.shape == (24, 64, 64, 1)
    assert targets["density"].shape == (24, 64, 64, 1)
    assert targets["gravity"].shape == (81, 81, 1)

    prediction = tf.Variable(tf.convert_to_tensor(migration_input[None, ...]))
    gravity = tf.convert_to_tensor(targets["gravity"][None, ...])
    density = tf.convert_to_tensor(targets["density"][None, ...])
    with tf.GradientTape() as tape:
        components = e02_loss_components(density, prediction, gravity)
    gradient = tape.gradient(components["total_loss"], prediction)
    for name in ("total_loss", "model_loss", "physics_loss", "struct_loss"):
        assert bool(tf.math.is_finite(components[name]).numpy()), name
    assert gradient is not None
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
