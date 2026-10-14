"""Differentiable TensorFlow Gz forward operator for experiment E02."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from physics.gravity_forward import CELL_SIZE, G, NX, NY, NZ, OBS_X, OBS_Y, OBS_Z

VOLUME_SHAPE = (NZ, NY, NX)
GRAVITY_SHAPE = (len(OBS_Y), len(OBS_X))
FFT_SIZE = 256
KERNEL_SIZE = 144
CONVOLUTION_CROP_START = 63


def _make_kernel_spectra() -> tf.Tensor:
    """Build padded correlation-kernel spectra as fixed TensorFlow constants."""
    kernel_indices = np.arange(KERNEL_SIZE, dtype=np.float64)
    # Observation index j and model index i have coordinate offset
    # cell-observation = 9 + i - j. Correlation lag t=j-i gives 9-t.
    coordinate_offsets = (9.0 - (kernel_indices - 63.0)) * CELL_SIZE
    dy, dx = np.meshgrid(coordinate_offsets, coordinate_offsets, indexing="ij")
    spectra = []
    for iz in range(NZ):
        depth = (iz + 0.5) * CELL_SIZE - OBS_Z
        radius_squared = dx**2 + dy**2 + depth**2
        kernel = G * 1e8 * CELL_SIZE**3 * depth / (radius_squared * np.sqrt(radius_squared))
        padded = np.zeros((FFT_SIZE, FFT_SIZE), dtype=np.float32)
        padded[:KERNEL_SIZE, :KERNEL_SIZE] = kernel.astype(np.float32)
        spectra.append(np.fft.fft2(padded).astype(np.complex64))
    return tf.constant(np.stack(spectra), dtype=tf.complex64, name="gz_kernel_spectra")


_KERNEL_SPECTRA = _make_kernel_spectra()


@tf.function(reduce_retracing=True, jit_compile=False)
def gravity_forward_tf(density: tf.Tensor) -> tf.Tensor:
    """Map density `[B,24,64,64,1]` to gravity `[B,81,81,1]` in mGal.

    This uses zero-padded 2D FFT correlations, one per depth slice. Padding to
    256 covers the full linear convolution size (64 + 144 - 1 = 207), so no
    periodic wrap-around enters the cropped receiver grid. All operations from
    density through predicted gravity are TensorFlow ops and remain differentiable.
    """
    density = tf.convert_to_tensor(density, dtype=tf.float32)
    tf.debugging.assert_rank(density, 5)
    tf.debugging.assert_equal(tf.shape(density)[1:4], VOLUME_SHAPE)
    tf.debugging.assert_equal(tf.shape(density)[4], 1)

    density = tf.squeeze(density, axis=-1)
    density = tf.pad(density, [[0, 0], [0, 0], [0, FFT_SIZE - NY], [0, FFT_SIZE - NX]])
    density_spectrum = tf.signal.fft2d(tf.cast(density, tf.complex64))
    convolved = tf.signal.ifft2d(
        density_spectrum * _KERNEL_SPECTRA[tf.newaxis, ...]
    )
    convolved = tf.math.real(convolved)
    start = CONVOLUTION_CROP_START
    gravity_by_depth = convolved[
        :, :, start:start + GRAVITY_SHAPE[0], start:start + GRAVITY_SHAPE[1]
    ]
    gravity = tf.reduce_sum(gravity_by_depth, axis=1)
    return gravity[..., tf.newaxis]
