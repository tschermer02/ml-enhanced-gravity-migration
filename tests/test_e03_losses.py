
"""
Tests for E03: Balanced MSE + Soft Tversky loss.

Run from the repository root:

    python -m pytest tests/test_e03_losses.py -v
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Make src/ importable when running tests directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gravity_migration_ml.ml.losses import balanced_density_mse

from gravity_migration_ml.ml.e03_losses import (
    e03_combined_loss,
    soft_tversky_loss,
    soft_occupancy,
    TVERSKY_WEIGHT,
    TVERSKY_ALPHA,
    TVERSKY_BETA,
    DENSITY_THRESHOLD,
)


# ============================================================
# 1. TEST COMBINED LOSS
# ============================================================

def test_combined_loss_formula():
    """Verify E03 = balanced MSE + 0.1 * Tversky."""

    truth = np.zeros((1, 4, 4, 4, 1), dtype=np.float32)
    truth[:, 1:3, 1:3, 1:3, :] = 0.5

    prediction = np.full_like(truth, 0.02)
    prediction[:, 1:3, 1:3, 1:3, :] = 0.4

    y_true = tf.constant(truth)
    y_pred = tf.constant(prediction)

    density_loss = balanced_density_mse(y_true, y_pred)
    geometry_loss = soft_tversky_loss(y_true, y_pred)
    combined = e03_combined_loss(y_true, y_pred)

    expected = density_loss + TVERSKY_WEIGHT * geometry_loss

    np.testing.assert_allclose(
        combined.numpy(),
        expected.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


# ============================================================
# 2. TEST CORRECT PREDICTIONS
# ============================================================

def test_perfect_prediction_has_lower_loss():
    """Correct density should score better than incorrect density."""

    truth = np.zeros((1, 8, 8, 8, 1), dtype=np.float32)
    truth[:, 2:6, 2:6, 2:6, :] = 0.8

    y_true = tf.constant(truth)

    correct_prediction = tf.constant(truth)
    incorrect_prediction = tf.ones_like(y_true) * 0.5

    correct_loss = e03_combined_loss(
        y_true, correct_prediction
    )

    incorrect_loss = e03_combined_loss(
        y_true, incorrect_prediction
    )

    assert correct_loss.numpy() < incorrect_loss.numpy()


# ============================================================
# 3. TEST FALSE-POSITIVE PENALTY
# ============================================================

def test_false_positive_penalty():
    """
    Test Tversky's asymmetric FP/FN weighting.

    Use a small body and compare predictions with equal
    numbers of false-positive and false-negative voxels.
    """

    # Build a balanced binary occupancy example.
    truth = np.zeros((1, 4, 4, 4, 1), dtype=np.float32)
    truth[:, 1:3, 1:3, 1:3, :] = 1.0

    false_positive = truth.copy()
    false_negative = truth.copy()

    # FP: add 4 voxels outside the true body.
    false_positive[:, 0, 0, 0:4, :] = 1.0

    # FN: remove 4 voxels inside the true body.
    false_negative[:, 1, 1, 1:3, :] = 0.0
    false_negative[:, 1, 2, 1:3, :] = 0.0

    assert TVERSKY_ALPHA > TVERSKY_BETA

    fp_loss = soft_tversky_loss(
        tf.constant(truth),
        tf.constant(false_positive),
    )

    fn_loss = soft_tversky_loss(
        tf.constant(truth),
        tf.constant(false_negative),
    )

    assert fp_loss.numpy() > fn_loss.numpy(), (
        "False positives should receive a higher "
        "Tversky penalty than false negatives."
    )


# ============================================================
# 4. TEST OCCUPANCY MAPPING
# ============================================================

def test_soft_occupancy():
    """Check that occupancy increases with density."""

    density = tf.constant(
        [0.0, DENSITY_THRESHOLD, 0.5, 1.0],
        dtype=tf.float32,
    )

    occupancy = soft_occupancy(density).numpy()

    assert np.all(np.diff(occupancy) > 0.0)

    np.testing.assert_allclose(
        occupancy[1], 0.5, atol=1e-6
    )

    assert np.all(occupancy >= 0.0)
    assert np.all(occupancy <= 1.0)


# ============================================================
# 5. TEST GRADIENT FLOW
# ============================================================

def test_finite_nonzero_gradients():
    """Verify differentiability for gradient-based training."""

    truth = np.zeros((1, 8, 8, 8, 1), dtype=np.float32)
    truth[:, 2:6, 2:6, 2:6, :] = 0.7

    y_true = tf.constant(truth)

    # Prediction deliberately differs from truth.
    prediction = tf.Variable(
        tf.fill(y_true.shape, 0.15)
    )

    with tf.GradientTape() as tape:
        loss = e03_combined_loss(y_true, prediction)

    gradients = tape.gradient(loss, prediction)

    assert gradients is not None
    assert np.all(np.isfinite(gradients.numpy()))
    assert np.any(np.abs(gradients.numpy()) > 0.0)


# ============================================================
# 6. TEST REAL MODEL DIMENSIONS
# ============================================================

def test_full_volume_shape():
    """
    Test the real E03 output dimensions:
    (batch, depth, northing, easting, channel).
    """

    shape = (2, 24, 64, 64, 1)

    truth = np.zeros(shape, dtype=np.float32)
    truth[:, 8:14, 20:36, 20:36, :] = 0.6

    prediction = np.full(shape, 0.02, dtype=np.float32)
    prediction[:, 8:14, 20:36, 20:36, :] = 0.5

    loss = e03_combined_loss(
        tf.constant(truth),
        tf.constant(prediction),
    )

    assert loss.shape == ()
    assert np.isfinite(loss.numpy())


# ============================================================
# 7. TEST ZERO-BODY EDGE CASE
# ============================================================

def test_empty_true_body():
    """Loss should remain finite for an empty density model."""

    truth = tf.zeros((1, 4, 4, 4, 1))
    prediction = tf.zeros_like(truth)

    loss = e03_combined_loss(truth, prediction)

    assert np.isfinite(loss.numpy())
    assert loss.numpy() >= 0.0
