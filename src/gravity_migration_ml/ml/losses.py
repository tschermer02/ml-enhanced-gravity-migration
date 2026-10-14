"""Density losses for migration-to-model reconstruction."""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def balanced_density_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Average body and background MSE so the zero background cannot dominate."""
    y_true = tf.cast(y_true, y_pred.dtype)
    body_mask = tf.cast(y_true > 0.0, y_pred.dtype)
    background_mask = 1.0 - body_mask
    squared_error = tf.square(y_pred - y_true)
    spatial_axes = tf.range(1, tf.rank(y_true))

    body_error = tf.math.divide_no_nan(
        tf.reduce_sum(squared_error * body_mask, axis=spatial_axes),
        tf.reduce_sum(body_mask, axis=spatial_axes) + tf.keras.backend.epsilon(),
    )
    background_error = tf.math.divide_no_nan(
        tf.reduce_sum(squared_error * background_mask, axis=spatial_axes),
        tf.reduce_sum(background_mask, axis=spatial_axes) + tf.keras.backend.epsilon(),
    )
    return tf.reduce_mean(0.5 * body_error + 0.5 * background_error)


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def ordinary_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Ordinary mean squared error baseline."""
    return tf.reduce_mean(tf.square(tf.cast(y_true, y_pred.dtype) - y_pred))
