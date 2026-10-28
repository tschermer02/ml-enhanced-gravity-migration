"""Reusable loss terms and weighted objectives for 3-D density inversion.

Arrays are (batch, z, y, x, channel), density is in g/cm^3.
The legacy occupancy options preserve differences in earlier experiments.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def ordinary_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - tf.cast(y_true, y_pred.dtype)))


def _huber_values(y_true, y_pred, delta=0.1):
    if delta <= 0:
        raise ValueError("delta must be positive")
    e = tf.abs(y_pred - tf.cast(y_true, y_pred.dtype))
    d = tf.cast(delta, y_pred.dtype)
    quadratic = tf.minimum(e, d)
    return 0.5 * tf.square(quadratic) + d * (e - quadratic)


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def huber_model_loss(y_true, y_pred, delta=0.1):
    return tf.reduce_mean(_huber_values(y_true, y_pred, delta))


def _balanced(values, y_true, signed=False):
    truth = tf.cast(y_true, values.dtype)
    mask = tf.abs(truth) > 0 if signed else truth > 0
    body = tf.cast(mask, values.dtype)
    background = 1.0 - body
    axes = tf.range(1, tf.rank(values))
    inside = tf.math.divide_no_nan(tf.reduce_sum(values * body, axis=axes),
                                   tf.reduce_sum(body, axis=axes))
    outside = tf.math.divide_no_nan(tf.reduce_sum(values * background, axis=axes),
                                    tf.reduce_sum(background, axis=axes))
    return tf.reduce_mean(0.5 * (inside + outside))


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def balanced_density_mse(y_true, y_pred, signed=False):
    values = tf.square(y_pred - tf.cast(y_true, y_pred.dtype))
    return _balanced(values, y_true, signed=signed)


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def balanced_huber_loss(y_true, y_pred, delta=0.1, signed=False):
    return _balanced(_huber_values(y_true, y_pred, delta), y_true, signed=signed)


def soft_occupancy(values, *, threshold=0.1, temperature=0.025,
                   signed=False, subtract_baseline=False):
    """Smooth body occupancy. Signed mode uses |density|.

    subtract_baseline=False reproduces the original Tversky mapping;
    True reproduces the baseline-subtracted vertical/signed mapping.
    """
    if temperature <= 0 or threshold < 0:
        raise ValueError("temperature must be > 0 and threshold >= 0")
    values = tf.abs(values) if signed else values
    t = tf.cast(threshold, values.dtype)
    temp = tf.cast(temperature, values.dtype)
    raw = tf.sigmoid((values - t) / temp)
    if subtract_baseline:
        baseline = tf.sigmoid(-t / temp)
        raw = tf.clip_by_value((raw - baseline) / (1.0 - baseline), 0.0, 1.0)
    return raw


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def soft_tversky_loss(y_true, y_pred, *, threshold=0.1, temperature=0.025,
                      alpha=0.7, beta=0.3, signed=False,
                      subtract_baseline=False):
    truth = tf.cast(y_true, y_pred.dtype)
    if signed:
        truth = tf.abs(truth)
    target = tf.cast(truth >= threshold, y_pred.dtype)
    pred = soft_occupancy(y_pred, threshold=threshold, temperature=temperature,
                          signed=signed, subtract_baseline=subtract_baseline)
    axes = tf.range(1, tf.rank(target))
    tp = tf.reduce_sum(target * pred, axis=axes)
    fp = tf.reduce_sum((1.0 - target) * pred, axis=axes)
    fn = tf.reduce_sum(target * (1.0 - pred), axis=axes)
    eps = tf.cast(1e-6, y_pred.dtype)
    return tf.reduce_mean(1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps))


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def vertical_profile_loss(y_true, y_pred, *, threshold=0.1,
                          temperature=0.025, signed=False):
    """Compare occupancy versus depth, center of mass, and vertical spread."""
    truth = tf.cast(y_true, y_pred.dtype)
    if signed:
        truth = tf.abs(truth)
    true_mask = tf.cast(truth >= threshold, y_pred.dtype)
    pred_mask = soft_occupancy(y_pred, threshold=threshold, temperature=temperature,
                               signed=signed, subtract_baseline=True)
    true = tf.reduce_sum(true_mask, axis=[2, 3, 4])
    pred = tf.reduce_sum(pred_mask, axis=[2, 3, 4])
    true = tf.math.divide_no_nan(true, tf.reduce_sum(true, axis=1, keepdims=True))
    pred = tf.math.divide_no_nan(pred, tf.reduce_sum(pred, axis=1, keepdims=True))
    profile_error = tf.reduce_sum(tf.abs(true - pred), axis=1)
    nz = tf.shape(y_pred)[1]
    depth = tf.cast(tf.range(nz), y_pred.dtype)[None, :] / tf.maximum(
        tf.cast(nz - 1, y_pred.dtype), 1.0)
    tc = tf.reduce_sum(true * depth, axis=1)
    pc = tf.reduce_sum(pred * depth, axis=1)
    tv = tf.reduce_sum(true * tf.square(depth - tc[:, None]), axis=1)
    pv = tf.reduce_sum(pred * tf.square(depth - pc[:, None]), axis=1)
    return tf.reduce_mean(profile_error + 0.5 * tf.abs(tc - pc) +
                          0.5 * tf.abs(tf.sqrt(tv + 1e-6) - tf.sqrt(pv + 1e-6)))


@tf.keras.utils.register_keras_serializable(package="gravity_migration_ml")
def structural_gradient_loss(y_true, y_pred, cell_spacing_m=10.0):
    if cell_spacing_m <= 0:
        raise ValueError("cell_spacing_m must be positive")
    truth = tf.cast(y_true, y_pred.dtype)
    spacing = tf.cast(cell_spacing_m, y_pred.dtype)
    losses = []
    for axis in (1, 2, 3):
        a = tf.experimental.numpy.diff(truth, axis=axis) / spacing
        b = tf.experimental.numpy.diff(y_pred, axis=axis) / spacing
        losses.append(tf.reduce_mean(tf.abs(a - b)))
    return tf.add_n(losses) / 3.0


def gravity_consistency_loss(density_pred, observed, forward_model: Callable,
                             normalization=3.5):
    """Physics loss; forward_model must be differentiable and match Gz/Gzz units."""
    if normalization <= 0:
        raise ValueError("normalization must be positive")
    calculated = forward_model(density_pred)
    observation = tf.cast(observed, calculated.dtype)
    return tf.reduce_mean(tf.square((calculated - observation) /
                                    tf.cast(normalization, calculated.dtype)))


def loss_components(y_true, y_pred, *, weights: Mapping[str, float] | None = None,
                    signed=False, threshold=0.1, temperature=0.025,
                    subtract_tversky_baseline=None, huber_delta=0.1,
                    cell_spacing_m=10.0, gravity_observed=None,
                    forward_model: Callable | None = None, gravity_normalization=3.5):
    """Compute only terms selected by nonzero weights and return their values.

    Allowed keys: mse, balanced_mse, huber, balanced_huber, tversky,
    depth, gradient, physics. ``total`` is their weighted sum.
    """
    weights = dict(weights or {"balanced_mse": 1.0})
    allowed = {"mse", "balanced_mse", "huber", "balanced_huber", "tversky",
               "depth", "gradient", "physics"}
    unknown = set(weights) - allowed
    if unknown:
        raise ValueError(f"Unknown loss terms: {sorted(unknown)}")
    if any(float(w) < 0 for w in weights.values()):
        raise ValueError("Loss weights must be nonnegative")
    if subtract_tversky_baseline is None:
        subtract_tversky_baseline = signed  # historical unsigned vs signed defaults
    result = {}
    for name, weight in weights.items():
        if float(weight) == 0:
            continue
        if name == "mse":
            value = ordinary_mse(y_true, y_pred)
        elif name == "balanced_mse":
            value = balanced_density_mse(y_true, y_pred, signed=signed)
        elif name == "huber":
            value = huber_model_loss(y_true, y_pred, delta=huber_delta)
        elif name == "balanced_huber":
            value = balanced_huber_loss(y_true, y_pred, delta=huber_delta, signed=signed)
        elif name == "tversky":
            value = soft_tversky_loss(y_true, y_pred, threshold=threshold,
                                      temperature=temperature, signed=signed,
                                      subtract_baseline=subtract_tversky_baseline)
        elif name == "depth":
            value = vertical_profile_loss(y_true, y_pred, threshold=threshold,
                                          temperature=temperature, signed=signed)
        elif name == "gradient":
            value = structural_gradient_loss(y_true, y_pred, cell_spacing_m)
        else:
            if gravity_observed is None or forward_model is None:
                raise ValueError("Physics loss requires gravity_observed and forward_model")
            value = gravity_consistency_loss(y_pred, gravity_observed, forward_model,
                                             gravity_normalization)
        result[name] = value
    result["total"] = (tf.add_n([tf.cast(weights[k], y_pred.dtype) * value
                                 for k, value in result.items()]) if result
                       else tf.zeros((), dtype=y_pred.dtype))
    return result


def make_loss(*, weights=None, **options):
    """Return a standard Keras loss callable (density-only terms).

    For physics loss use loss_components within a custom training step, because
    observed gravity is not available to a standard two-argument Keras loss.
    """
    if weights is not None and float(weights.get("physics", 0)) != 0:
        raise ValueError("Physics loss requires a custom training step")

    def combined_loss(y_true, y_pred):
        return loss_components(y_true, y_pred, weights=weights, **options)["total"]

    combined_loss.__name__ = "combined_density_loss"
    return combined_loss
