
"""
E03 loss functions for 3D gravity migration reconstruction.

E01: Balanced density MSE
E02: Balanced Huber loss
E03: Balanced density MSE + Soft Tversky loss

The E03 objective combines density accuracy with body geometry.
"""

import tensorflow as tf

from gravity_migration_ml.ml.losses import balanced_density_mse


# ============================================================
# 1. E03 PARAMETERS
# ============================================================

TVERSKY_WEIGHT = 0.1

TVERSKY_ALPHA = 0.7  # False-positive penalty
TVERSKY_BETA = 0.3   # False-negative penalty

DENSITY_THRESHOLD = 0.1  # g/cm^3

# Width of smooth occupancy transition (g/cm^3)
OCCUPANCY_TEMPERATURE = 0.025

EPSILON = 1e-6


# ============================================================
# 2. SOFT OCCUPANCY MAPPING
# ============================================================

def soft_occupancy(y_pred):
    """
    Convert continuous density predictions into soft body
    occupancy probabilities.

    Uses a smooth sigmoid centered on the density threshold.

    At predicted density = 0.1 g/cm^3:
        occupancy = 0.5

    Returns values between 0 and 1.
    """
    threshold = tf.cast(DENSITY_THRESHOLD, y_pred.dtype)
    temperature = tf.cast(OCCUPANCY_TEMPERATURE, y_pred.dtype)

    return tf.sigmoid((y_pred - threshold) / temperature)


# ============================================================
# 3. SOFT TVERSKY LOSS
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="gravity_migration_ml"
)
def soft_tversky_loss(y_true, y_pred):
    """
    Differentiable Tversky loss for 3D body reconstruction.

    Alpha = 0.7: penalize false positives.
    Beta  = 0.3: penalize false negatives.

    The true density defines a binary body mask.
    The predicted density is converted to soft occupancy.

    Computes a Tversky score per sample, then averages
    across the batch.
    """
    y_true = tf.cast(y_true, y_pred.dtype)

    threshold = tf.cast(DENSITY_THRESHOLD, y_pred.dtype)
    alpha = tf.cast(TVERSKY_ALPHA, y_pred.dtype)
    beta = tf.cast(TVERSKY_BETA, y_pred.dtype)
    epsilon = tf.cast(EPSILON, y_pred.dtype)

    true_body = tf.cast(y_true >= threshold, y_pred.dtype)
    predicted_body = soft_occupancy(y_pred)

    # Sum over depth, northing, easting, and channel
    # while preserving the batch dimension.
    axes = tf.range(1, tf.rank(y_true))

    true_positive = tf.reduce_sum(
        true_body * predicted_body,
        axis=axes
    )

    false_positive = tf.reduce_sum(
        (1.0 - true_body) * predicted_body,
        axis=axes
    )

    false_negative = tf.reduce_sum(
        true_body * (1.0 - predicted_body),
        axis=axes
    )

    tversky_score = (
        (true_positive + epsilon)
        /
        (
            true_positive
            + alpha * false_positive
            + beta * false_negative
            + epsilon
        )
    )

    return tf.reduce_mean(1.0 - tversky_score)


# ============================================================
# 4. COMBINED E03 LOSS
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="gravity_migration_ml"
)
def e03_combined_loss(y_true, y_pred):
    """
    E03: Balanced MSE + 0.1 * Soft Tversky.

    Balanced MSE:
        Preserves density accuracy inside and outside bodies.

    Soft Tversky:
        Encourages accurate body support and penalizes
        false-positive density predictions.
    """
    density_loss = balanced_density_mse(y_true, y_pred)

    geometry_loss = soft_tversky_loss(y_true, y_pred)

    return density_loss + TVERSKY_WEIGHT * geometry_loss
