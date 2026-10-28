import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from src.losses.losses import (balanced_density_mse, make_loss, loss_components,
                               soft_tversky_loss, vertical_profile_loss)


def test_signed_balancing_detects_negative_body():
    true = tf.constant([[[[[-0.5], [0.0]]]]], dtype=tf.float32)
    pred = tf.zeros_like(true)
    assert np.isclose(balanced_density_mse(true, pred, signed=True).numpy(), 0.125)
    assert np.isclose(balanced_density_mse(true, pred, signed=False).numpy(), 0.0625)


def test_composition_and_gradient():
    truth = tf.zeros((2, 4, 4, 4, 1), dtype=tf.float32)
    pred = tf.zeros_like(truth)
    loss = make_loss(weights={"balanced_mse": 1, "tversky": 0.1, "depth": 0.02},
                     signed=True)
    assert np.isfinite(loss(truth, pred).numpy())
    components = loss_components(truth, pred, weights={"mse": 1, "gradient": 0.5})
    assert np.isclose(components["total"].numpy(), 0)


def test_physics_requires_observations():
    truth = tf.zeros((1, 4, 4, 4, 1), dtype=tf.float32)
    with pytest.raises(ValueError):
        loss_components(truth, truth, weights={"physics": 1})
