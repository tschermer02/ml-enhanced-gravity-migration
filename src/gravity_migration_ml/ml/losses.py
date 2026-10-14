
"""
Loss functions for 3D density reconstruction.

E01: Balanced MSE
E02: Balanced Huber
"""

import tensorflow as tf


# ============================================================
# E01: BALANCED DENSITY MSE
# ============================================================

@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def balanced_density_mse(y_true, y_pred):
    """
    Balanced mean squared error between body and background.
    """

    y_true = tf.cast(y_true, y_pred.dtype)

    body_mask = tf.cast(y_true > 0.0, y_pred.dtype)
    background_mask = 1.0 - body_mask

    squared_error = tf.square(y_pred - y_true)

    axes = tf.range(1, tf.rank(y_true))

    body_loss = tf.math.divide_no_nan(
        tf.reduce_sum(squared_error * body_mask, axis=axes),
        tf.reduce_sum(body_mask, axis=axes)
    )

    background_loss = tf.math.divide_no_nan(
        tf.reduce_sum(squared_error * background_mask, axis=axes),
        tf.reduce_sum(background_mask, axis=axes)
    )

    return tf.reduce_mean(
        0.5 * body_loss + 0.5 * background_loss
    )


# ============================================================
# E02: BALANCED HUBER LOSS
# ============================================================

@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def balanced_huber_loss(y_true, y_pred):
    """
    Balanced Huber loss for 3D density reconstruction.

    Body and background are weighted equally.

    Huber delta = 0.1 g/cm^3
    """

    y_true = tf.cast(y_true, y_pred.dtype)

    delta = tf.cast(0.1, y_pred.dtype)

    # Body and background masks
    body_mask = tf.cast(y_true > 0.0, y_pred.dtype)
    background_mask = 1.0 - body_mask

    # Absolute density error
    error = tf.abs(y_pred - y_true)

    # Elementwise Huber loss
    quadratic = tf.minimum(error, delta)
    linear = error - quadratic

    huber_error = (
        0.5 * tf.square(quadratic)
        + delta * linear
    )

    # Spatial and channel dimensions
    axes = tf.range(1, tf.rank(y_true))

    # Mean error inside the true body
    body_loss = tf.math.divide_no_nan(
        tf.reduce_sum(huber_error * body_mask, axis=axes),
        tf.reduce_sum(body_mask, axis=axes)
    )

    # Mean error in the background
    background_loss = tf.math.divide_no_nan(
        tf.reduce_sum(huber_error * background_mask, axis=axes),
        tf.reduce_sum(background_mask, axis=axes)
    )

    # Equal contribution from body and background
    total_loss = (
        0.5 * body_loss
        + 0.5 * background_loss
    )

    return tf.reduce_mean(total_loss)


# ============================================================
# ORDINARY MSE BASELINE
# ============================================================

@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def ordinary_mse(y_true, y_pred):
    """Ordinary mean squared error."""

    y_true = tf.cast(y_true, y_pred.dtype)

    return tf.reduce_mean(
        tf.square(y_pred - y_true)
    )
