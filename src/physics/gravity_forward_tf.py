"""Differentiable gravity forward models for physics-guided training.

`gravity_forward_tf` preserves the original Gz TensorFlow implementation.
`gravity_gzz_forward_tf` adds Gzz (Eotvos) for E11C/E11E, with anisotropic
cell dimensions and an observation height that can vary across a batch.

Density: (batch, 24, 64, 64, 1), in g/cm^3.
Gz:      (batch, 81, 81, 1), in mGal.
Gzz:     (batch, 81, 81, 1), in Eotvos.

The Gzz calculation matches the project's *cell-centered point-mass*
NumPy/SciPy FFT forward model, not an exact rectangular-prism integral.
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


# ---------------------------------------------------------------------------
# Legacy Gz implementation: preserved as-is for older experiments.
# ---------------------------------------------------------------------------
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
    """Original differentiable Gz forward model (mGal; legacy 10 m grid)."""
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


# ---------------------------------------------------------------------------
# E11 Gzz: same FFT convolution convention as gravity_forward.calculate_gzz.
# ---------------------------------------------------------------------------
def _per_sample(values, batch_size, name):
    """Convert a scalar or a length-B vector to a float32 vector of length B."""
    values = tf.reshape(tf.convert_to_tensor(values, dtype=tf.float32), [-1])
    n = tf.size(values)
    tf.debugging.assert_equal(
        tf.logical_or(tf.equal(n, 1), tf.equal(n, batch_size)), True,
        message=f"{name} must be a scalar or one value per sample",
    )
    return tf.broadcast_to(values, [batch_size])


@tf.function(reduce_retracing=True, jit_compile=False)
def gravity_gzz_forward_tf(
    density,
    observation_height_m=0.0,
    dx_m=CELL_SIZE,
    dy_m=CELL_SIZE,
    dz_m=CELL_SIZE,
):
    """Differentiable Gzz in Eotvos with per-sample geometry and height.

    Parameters may be scalars or length-B arrays/tensors, where B is the
    batch dimension. z is positive downward; observation_height_m is >= 0
    above the model surface. All lengths are in meters.

    The observation grid uses the same fixed relative alignment as the NumPy
    model: (81,81) cells around the (64,64) model grid.
    """
    density = tf.convert_to_tensor(density, dtype=tf.float32)
    tf.debugging.assert_rank(density, 5)
    tf.debugging.assert_equal(tf.shape(density)[1:4], VOLUME_SHAPE)
    tf.debugging.assert_equal(tf.shape(density)[4], 1)
    tf.debugging.assert_all_finite(density, "Density contains NaN/Inf")

    batch_size = tf.shape(density)[0]
    height = _per_sample(observation_height_m, batch_size, "observation_height_m")
    dx = _per_sample(dx_m, batch_size, "dx_m")
    dy = _per_sample(dy_m, batch_size, "dy_m")
    dz = _per_sample(dz_m, batch_size, "dz_m")

    for name, value in (("dx_m", dx), ("dy_m", dy), ("dz_m", dz),
                        ("observation_height_m", height)):
        tf.debugging.assert_all_finite(value, f"{name} contains NaN/Inf")
    tf.debugging.assert_positive(dx)
    tf.debugging.assert_positive(dy)
    tf.debugging.assert_positive(dz)
    tf.debugging.assert_greater_equal(height, tf.zeros_like(height))

    # Kernel shape: (batch, depth layer, offset-y, offset-x).
    # NumPy's gzz_kernel uses offsets [-72, ..., 71] for each axis.
    offsets = tf.cast(tf.range(-72, 72), tf.float32)
    oy, ox = tf.meshgrid(offsets, offsets, indexing="ij")
    x = ox[None, None, :, :] * dx[:, None, None, None]
    y = oy[None, None, :, :] * dy[:, None, None, None]
    z_index = tf.cast(tf.range(NZ), tf.float32) + 0.5
    separation = (z_index[None, :] * dz[:, None] + height[:, None])[:, :, None, None]

    r2 = x*x + y*y + separation*separation
    kernel = (3.0 * separation*separation - r2) / (r2*r2*tf.sqrt(r2))

    # FFT padding replicates scipy.signal.fftconvolve(layer, kernel, 'full').
    # Full linear convolution is (64+144-1) = 207, less than FFT_SIZE 256.
    kernel = tf.pad(kernel, [[0, 0], [0, 0],
                                    [0, FFT_SIZE-KERNEL_SIZE],
                                    [0, FFT_SIZE-KERNEL_SIZE]])
    layers = tf.squeeze(density, axis=-1)
    layers = tf.pad(layers, [[0, 0], [0, 0],
                                     [0, FFT_SIZE-NY], [0, FFT_SIZE-NX]])

    layer_fft = tf.signal.fft2d(tf.cast(layers, tf.complex64))
    kernel_fft = tf.signal.fft2d(tf.cast(kernel, tf.complex64))
    # Sum linear contributions across depth before the inverse FFT.
    field_fft = tf.reduce_sum(layer_fft * kernel_fft, axis=1)
    full_field = tf.math.real(tf.signal.ifft2d(field_fft))
    field = full_field[:, CROP_START:CROP_START+GRAVITY_SHAPE[0],
                       CROP_START:CROP_START+GRAVITY_SHAPE[1]]

    # density g/cm^3 -> kg/m^3 (*1000), SI gradient -> Eotvos (*1e9).
    multiplier = G * 1e12 * dx * dy * dz
    return (field * multiplier[:, None, None])[..., None]
