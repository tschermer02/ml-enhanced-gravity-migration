"""Custom Keras training step for E02's physics-constrained objective."""
from __future__ import annotations

import tensorflow as tf

from gravity_migration_ml.ml.e02_losses import e02_loss_components


class E02Trainer(tf.keras.Model):
    """Train the unchanged E01 network with density, gravity, and gradient losses."""

    def __init__(self, network: tf.keras.Model, loss_weights: dict[str, float],
                 huber_delta: float = 0.1,
                 gravity_normalization_mgal: float = 0.02,
                 cell_spacing_m: float = 10.0):
        super().__init__(name="e02_physics_constrained_trainer")
        self.network = network
        self.loss_weights = dict(loss_weights)
        self.huber_delta = float(huber_delta)
        self.gravity_normalization_mgal = float(gravity_normalization_mgal)
        self.cell_spacing_m = float(cell_spacing_m)
        self._trackers = {
            name: tf.keras.metrics.Mean(name=name)
            for name in ("total_loss", "model_loss", "physics_loss", "struct_loss")
        }

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return list(self._trackers.values())

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.network(inputs, training=training)

    def _components(self, inputs: tf.Tensor, targets: dict[str, tf.Tensor],
                    training: bool) -> dict[str, tf.Tensor]:
        density_prediction = self.network(inputs, training=training)
        return e02_loss_components(
            targets["density"], density_prediction, targets["gravity"],
            **self.loss_weights,
            huber_delta=self.huber_delta,
            gravity_normalization_mgal=self.gravity_normalization_mgal,
            cell_spacing_m=self.cell_spacing_m,
        )

    def train_step(self, data: tuple[tf.Tensor, dict[str, tf.Tensor]]) -> dict[str, tf.Tensor]:
        inputs, targets = data
        with tf.GradientTape() as tape:
            components = self._components(inputs, targets, training=True)
        gradients = tape.gradient(components["total_loss"], self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        for name, tracker in self._trackers.items():
            tracker.update_state(components[name])
        return {name: tracker.result() for name, tracker in self._trackers.items()}

    def test_step(self, data: tuple[tf.Tensor, dict[str, tf.Tensor]]) -> dict[str, tf.Tensor]:
        inputs, targets = data
        components = self._components(inputs, targets, training=False)
        for name, tracker in self._trackers.items():
            tracker.update_state(components[name])
        return {name: tracker.result() for name, tracker in self._trackers.items()}
