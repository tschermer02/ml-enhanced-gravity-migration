"""Physics-constrained loss components for experiment E02."""
from __future__ import annotations

import tensorflow as tf

from gravity_migration_ml.ml.e02_physics import gravity_forward_tf


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def huber_model_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
                     delta: float = 0.1) -> tf.Tensor:
    """Mean Huber error in density units (g/cm^3)."""
    delta_tensor = tf.cast(delta, y_pred.dtype)
    error = tf.abs(tf.cast(y_true, y_pred.dtype) - y_pred)
    quadratic = tf.minimum(error, delta_tensor)
    linear = error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta_tensor * linear)


def structural_gradient_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
                             cell_spacing_m: float = 10.0) -> tf.Tensor:
    """Mean L1 discrepancy of x, y, z finite differences, in density/m."""
    y_true = tf.cast(y_true, y_pred.dtype)
    spacing = tf.cast(cell_spacing_m, y_pred.dtype)
    losses = []
    for axis in (3, 2, 1):  # x, y, z for [batch, z, y, x, channel]
        predicted_gradient = tf.experimental.numpy.diff(y_pred, axis=axis) / spacing
        true_gradient = tf.experimental.numpy.diff(y_true, axis=axis) / spacing
        losses.append(tf.reduce_mean(tf.abs(predicted_gradient - true_gradient)))
    return tf.add_n(losses) / tf.cast(3.0, y_pred.dtype)


def e02_loss_components(
    density_true: tf.Tensor,
    density_predicted: tf.Tensor,
    gravity_observed: tf.Tensor,
    *,
    lambda_model: float = 1.0,
    lambda_phys: float = 0.1,
    lambda_struct: float = 0.1,
    lambda_reg: float = 0.0,
    huber_delta: float = 0.1,
    gravity_normalization_mgal: float = 3.5,
    cell_spacing_m: float = 10.0,
) -> dict[str, tf.Tensor]:
    """Compute E02's weighted Huber, gravity, structural, and zero reg terms.

    Gravity normalization scales only the residual used by this loss. The
    differentiable forward operator itself always returns physical mGal.
    """
    if gravity_normalization_mgal <= 0:
        raise ValueError("gravity_normalization_mgal must be positive")
    density_true = tf.cast(density_true, density_predicted.dtype)
    gravity_observed = tf.cast(gravity_observed, density_predicted.dtype)
    gravity_predicted = gravity_forward_tf(density_predicted)
    gravity_residual = (
        gravity_predicted - gravity_observed
    ) / tf.cast(gravity_normalization_mgal, density_predicted.dtype)

    model_loss = huber_model_loss(density_true, density_predicted, huber_delta)
    physics_loss = tf.reduce_mean(tf.square(gravity_residual))
    struct_loss = structural_gradient_loss(
        density_true, density_predicted, cell_spacing_m,
    )
    regularization_loss = tf.zeros((), dtype=density_predicted.dtype)
    total_loss = (
        tf.cast(lambda_model, density_predicted.dtype) * model_loss
        + tf.cast(lambda_phys, density_predicted.dtype) * physics_loss
        + tf.cast(lambda_struct, density_predicted.dtype) * struct_loss
        + tf.cast(lambda_reg, density_predicted.dtype) * regularization_loss
    )
    return {
        "total_loss": total_loss,
        "model_loss": model_loss,
        "physics_loss": physics_loss,
        "struct_loss": struct_loss,
        "regularization_loss": regularization_loss,
        "gravity_predicted": gravity_predicted,
    }
