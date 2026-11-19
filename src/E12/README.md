"""Differentiable independent E12 FFT Gzz forward model.

Stores exact staggered-grid transfer functions for configured receiver heights.
No calls to E11 or common physics. Requires TensorFlow only on import.
"""
from functools import lru_cache
import numpy as np
import tensorflow as tf
from .config import SHAPE, OBS_SHAPE, FFT_N, HEIGHTS_VARIABLE
from .physics import kernel_fft, _MODEL_INDICES, _OBS_INDICES, _MULTIPLIER

@lru_cache(maxsize=2)
def _cache(heights):
    return tf.constant(np.stack([kernel_fft(float(h)) for h in heights],axis=0),tf.complex64)

@lru_cache(maxsize=1)
def _index():
    z,y,x=np.indices(SHAPE,dtype=np.int32)
    indices=np.stack((z,(2+5*y),(2+5*x)),axis=-1).reshape(-1,3)
    return tf.constant(indices,tf.int32)


def forward(density, heights_m, allowed_heights=HEIGHTS_VARIABLE):
    """density [B,24,64,64,1] -> Gzz [B,81,81,1] (Eotvos)."""
    density=tf.convert_to_tensor(density,tf.float32)
    tf.debugging.assert_equal(tf.shape(density)[1:],(24,64,64,1))
    heights_m=tf.reshape(tf.cast(heights_m,tf.float32),[-1])
    b=tf.shape(density)[0]
    tf.debugging.assert_equal(tf.size(heights_m),b)
    options=tf.constant(allowed_heights,tf.float32)
    distance=tf.abs(heights_m[:,None]-options[None,:])
    tf.debugging.assert_less_equal(tf.reduce_min(distance,axis=1),.001,
        message='Physics height not among configured discrete heights')
    sel=tf.argmin(distance,axis=1,output_type=tf.int32)
    idx=_index()
    per=tf.shape(idx)[0]
    repeats=tf.tile(idx[None,:,:],[b,1,1])
    batch_ids=tf.tile(tf.range(b,dtype=tf.int32)[:,None,None],[1,per,1])
    all_ids=tf.reshape(tf.concat([batch_ids,repeats],axis=-1),[-1,4])
    updates=tf.reshape(density,[-1])
    sparse=tf.scatter_nd(all_ids,updates,[b,24,321,321])
    sparse=tf.pad(sparse,[[0,0],[0,0],[0,703],[0,703]])
    fourier=tf.signal.rfft2d(sparse)
    kernels=tf.gather(_cache(tuple(float(h) for h in allowed_heights)),sel)
    spectrum=tf.reduce_sum(fourier*kernels,axis=1)
    full=tf.signal.irfft2d(spectrum,fft_length=[FFT_N,FFT_N])
    targets=tf.constant((320+_OBS_INDICES).astype(np.int32))
    grid=tf.gather(tf.gather(full,targets,axis=1),targets,axis=2)
    return (grid*tf.constant(_MULTIPLIER,tf.float32))[...,None]
