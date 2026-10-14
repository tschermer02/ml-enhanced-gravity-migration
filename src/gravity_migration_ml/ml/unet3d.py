"""Small 3D U-Net for mapping gravity migration volumes to density."""
from __future__ import annotations

import tensorflow as tf


def _conv_block(values: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Apply two same-padded 3D convolutions with ReLU activation."""
    values = tf.keras.layers.Conv3D(
        filters, kernel_size=3, padding="same", activation="relu",
        name=f"{name}_conv1",
    )(values)
    return tf.keras.layers.Conv3D(
        filters, kernel_size=3, padding="same", activation="relu",
        name=f"{name}_conv2",
    )(values)


def build_unet3d(input_shape: tuple[int, int, int, int] = (24, 64, 64, 1)) -> tf.keras.Model:
    """Build a compact 3D U-Net with a one-channel sigmoid density output."""
    inputs = tf.keras.Input(shape=input_shape, name="migration_input")

    encoder_one = _conv_block(inputs, 16, "encoder1")
    pooled_one = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), name="pool1")(encoder_one)

    encoder_two = _conv_block(pooled_one, 32, "encoder2")
    pooled_two = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), name="pool2")(encoder_two)

    bottleneck = _conv_block(pooled_two, 64, "bottleneck")

    upsample_two = tf.keras.layers.UpSampling3D(size=(2, 2, 2), name="upsample2")(bottleneck)
    decoder_two = tf.keras.layers.Concatenate(name="skip2")([upsample_two, encoder_two])
    decoder_two = _conv_block(decoder_two, 32, "decoder2")

    upsample_one = tf.keras.layers.UpSampling3D(size=(2, 2, 2), name="upsample1")(decoder_two)
    decoder_one = tf.keras.layers.Concatenate(name="skip1")([upsample_one, encoder_one])
    decoder_one = _conv_block(decoder_one, 16, "decoder1")

    outputs = tf.keras.layers.Conv3D(
        1, kernel_size=1, activation="sigmoid", name="density_prediction",
    )(decoder_one)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="gravity_migration_unet3d")
