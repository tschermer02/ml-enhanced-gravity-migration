"""E05 vertical depth-profile loss for volumes shaped (batch,z,y,x,channel).

E05 = E04 original balanced-MSE + Tversky loss + 0.02 * depth-profile loss.
True and predicted occupancy are aggregated over X/Y to compare vertical
distributions, center depth, and depth spread for each sample.
"""
import tensorflow as tf
from gravity_migration_ml.ml.e03_losses import e03_combined_loss, DENSITY_THRESHOLD, OCCUPANCY_TEMPERATURE

DEPTH_WEIGHT = 0.02
EPSILON = 1e-6

@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def vertical_profile_loss(y_true, y_pred):
    """Compare vertical occupancy profiles, centers, and depth spread."""
    dtype = y_pred.dtype
    y_true = tf.cast(y_true, dtype)
    threshold = tf.cast(DENSITY_THRESHOLD, dtype)
    temperature = tf.cast(OCCUPANCY_TEMPERATURE, dtype)

    true_mask = tf.cast(y_true >= threshold, dtype)
    # Subtract occupancy's background response so zero prediction maps to zero.
    baseline = tf.sigmoid(-threshold / temperature)
    soft_mask = tf.sigmoid((y_pred - threshold) / temperature)
    pred_mask = tf.clip_by_value((soft_mask - baseline) / (1.0 - baseline), 0.0, 1.0)

    true_profile = tf.reduce_sum(true_mask, axis=[2, 3, 4])
    pred_profile = tf.reduce_sum(pred_mask, axis=[2, 3, 4])
    true_profile = tf.math.divide_no_nan(true_profile, tf.reduce_sum(true_profile, axis=1, keepdims=True))
    pred_profile = tf.math.divide_no_nan(pred_profile, tf.reduce_sum(pred_profile, axis=1, keepdims=True))

    profile_error = tf.reduce_sum(tf.abs(true_profile - pred_profile), axis=1)
    nz = tf.shape(y_pred)[1]
    depth = tf.cast(tf.range(nz), dtype)[tf.newaxis, :] / tf.maximum(tf.cast(nz - 1, dtype), 1.0)
    true_center = tf.reduce_sum(true_profile * depth, axis=1)
    pred_center = tf.reduce_sum(pred_profile * depth, axis=1)
    center_error = tf.abs(true_center - pred_center)

    true_var = tf.reduce_sum(true_profile * tf.square(depth - true_center[:, None]), axis=1)
    pred_var = tf.reduce_sum(pred_profile * tf.square(depth - pred_center[:, None]), axis=1)
    spread_error = tf.abs(tf.sqrt(true_var + EPSILON) - tf.sqrt(pred_var + EPSILON))
    return tf.reduce_mean(profile_error + 0.5 * center_error + 0.5 * spread_error)

@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def e05_combined_loss(y_true, y_pred):
    """E05 combined loss; E04 objective is unchanged except added depth term."""
    return e03_combined_loss(y_true, y_pred) + DEPTH_WEIGHT * vertical_profile_loss(y_true, y_pred)
