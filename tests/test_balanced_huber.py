"""Tests for the balanced body/background Huber loss used by E02."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from gravity_migration_ml.ml.losses import balanced_density_mse, balanced_huber_loss
from gravity_migration_ml.ml.unet3d import build_unet3d


def test_perfect_prediction_has_zero_loss() -> None:
    truth = tf.constant([[[[[0.0], [0.4]], [[0.0], [0.8]]]]], dtype=tf.float32)
    assert float(balanced_huber_loss(truth, truth).numpy()) == 0.0


def test_background_only_prediction_penalizes_true_body() -> None:
    truth = tf.zeros((2, 24, 64, 64, 1), dtype=tf.float32)
    truth = tf.tensor_scatter_nd_update(
        truth,
        indices=[[0, 4, 20, 25, 0], [1, 8, 30, 35, 0]],
        updates=[0.6, 0.8],
    )
    prediction = tf.zeros_like(truth)
    huber_loss = balanced_huber_loss(truth, prediction)
    mse_loss = balanced_density_mse(truth, prediction)
    assert float(huber_loss.numpy()) > 0.0
    assert float(mse_loss.numpy()) > 0.0
    assert np.isfinite(float(huber_loss.numpy()))


def test_balanced_huber_gradient_is_finite() -> None:
    truth = tf.zeros((2, 24, 64, 64, 1), dtype=tf.float32)
    truth = tf.tensor_scatter_nd_update(truth, [[0, 2, 3, 4, 0]], [0.7])
    prediction = tf.Variable(tf.ones_like(truth) * 0.25)
    with tf.GradientTape() as tape:
        loss = balanced_huber_loss(truth, prediction)
    gradient = tape.gradient(loss, prediction)
    assert gradient is not None
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
    assert float(tf.reduce_max(tf.abs(gradient)).numpy()) > 0.0


def test_balanced_huber_and_mse_are_finite_on_same_tensors() -> None:
    rng = np.random.default_rng(42)
    truth = np.zeros((2, 24, 64, 64, 1), dtype=np.float32)
    truth[:, 3:7, 20:28, 30:38, :] = 0.65
    prediction = rng.uniform(0, 1, size=truth.shape).astype(np.float32)
    huber = float(balanced_huber_loss(truth, prediction).numpy())
    mse = float(balanced_density_mse(truth, prediction).numpy())
    assert np.isfinite(huber) and np.isfinite(mse)
    assert huber != mse


def test_masked_huber_means_are_per_sample() -> None:
    truth = tf.constant([[[[[0.0], [0.2]]]], [[[[0.0], [0.0]]]]], dtype=tf.float32)
    prediction = tf.constant([[[[[0.2], [0.0]]]], [[[[0.0], [0.2]]]]], dtype=tf.float32)
    actual = float(balanced_huber_loss(truth, prediction).numpy())
    # Sample one has body/background Huber means both equal H_0.1(0.2)=0.015.
    # Sample two has no body, so its body mean is safely zero and its
    # background mean is (H(0)+H(0.2))/2 = 0.0075.
    expected = (0.015 + 0.00375) / 2
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_balanced_huber_keras_serialization_round_trip() -> None:
    serialized = tf.keras.losses.serialize(balanced_huber_loss)
    restored = tf.keras.losses.deserialize(serialized)
    truth = tf.zeros((1, 2, 2, 2, 1), dtype=tf.float32)
    prediction = tf.ones_like(truth) * 0.2
    assert np.isfinite(float(restored(truth, prediction).numpy()))


def test_unet_output_shape_is_unchanged() -> None:
    model = build_unet3d()
    assert model.output_shape == (None, 24, 64, 64, 1)
