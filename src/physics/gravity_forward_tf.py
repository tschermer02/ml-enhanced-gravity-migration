"""Differentiable TensorFlow Gz forward model for physics-guided training.

Kept separate because it is a TensorFlow/FFT implementation, not the NumPy
reference operator. This implementation currently supports Gz only.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from .gravity_forward import CELL_SIZE, G, NX, NY, NZ, OBS_X, OBS_Y, OBS_Z

VOLUME_SHAPE = (NZ, NY, NX)
GRAVITY_SHAPE = (len(OBS_Y), len(OBS_X))
FFT_SIZE = 256
KERNEL_SIZE = 144
CROP_START = 63


def _make_kernel_spectra():
    indices = np.arange(KERNEL_SIZE, dtype=np.float64)
    offsets = (9.0 - (indices - 63.0)) * CELL_SIZE
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    spectra = []
    for iz in range(NZ):
        depth = (iz + .5) * CELL_SIZE - OBS_Z
        r2 = dx**2 + dy**2 + depth**2
        kernel = G * 1e8 * CELL_SIZE**3 * depth / (r2 * np.sqrt(r2))
        padded = np.zeros((FFT_SIZE, FFT_SIZE), dtype=np.float32)
        padded[:KERNEL_SIZE, :KERNEL_SIZE] = kernel.astype(np.float32)
        spectra.append(np.fft.fft2(padded).astype(np.complex64))
    return tf.constant(np.stack(spectra), dtype=tf.complex64)

_KERNEL_SPECTRA = _make_kernel_spectra()


@tf.function(reduce_retracing=True, jit_compile=False)
def gravity_forward_tf(density):
    density = tf.convert_to_tensor(density, dtype=tf.float32)
    tf.debugging.assert_rank(density, 5)
    tf.debugging.assert_equal(tf.shape(density)[1:4], VOLUME_SHAPE)
    tf.debugging.assert_equal(tf.shape(density)[4], 1)
    density = tf.squeeze(density, axis=-1)
    density = tf.pad(density, [[0, 0], [0, 0], [0, FFT_SIZE-NY], [0, FFT_SIZE-NX]])
    spectrum = tf.signal.fft2d(tf.cast(density, tf.complex64))
    result = tf.math.real(tf.signal.ifft2d(spectrum * _KERNEL_SPECTRA[tf.newaxis, ...]))
    gravity = tf.reduce_sum(result[:, :, CROP_START:CROP_START+GRAVITY_SHAPE[0], CROP_START:CROP_START+GRAVITY_SHAPE[1]], axis=1)
    return gravity[..., tf.newaxis]
