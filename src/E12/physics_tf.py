"""Independent E12 Gzz forward and depth-weighted migration operators.

Exact point-mass cell-center kernel evaluated on *different* observation/model
coordinates. FFT acceleration uses a 25 m lattice, without spatial
interpolation of either density or observed Gzz. The 12.5 m model-center
stagger is represented explicitly in the kernel.

Density: g/cm^3. Gzz: Eotvos. z positive downward. Observation-plane
height h above z=0 is positive. No receiver margin is used.
"""
from functools import lru_cache
import numpy as np
from scipy.fft import rfft2, irfft2, next_fast_len
from .config import (SHAPE, OBS_SHAPE, MODEL_DX_M, MODEL_DY_M,
                     MODEL_DZ_M, OBS_DX_M, OBS_DY_M, GRID_STEP_M,
                     FFT_N, KERNEL_RADIUS)
G = 6.67e-11
# Coordinates (meters, relative to SW survey corner):
OBS_X = np.arange(81, dtype=np.float64)*OBS_DX_M
OBS_Y = np.arange(81, dtype=np.float64)*OBS_DY_M
MODEL_X = (np.arange(64, dtype=np.float64)+0.5)*MODEL_DX_M
MODEL_Y = (np.arange(64, dtype=np.float64)+0.5)*MODEL_DY_M
MODEL_Z = (np.arange(24, dtype=np.float64)+0.5)*MODEL_DZ_M
# Sparse-plane coordinates are 25*(2+5*j) = 50+125*j,
# actual model centers are 62.5+125*j => 12.5 m offset in both axes.
_MODEL_INDICES = 2 + 5*np.arange(64)
_OBS_INDICES = 4*np.arange(81)
_SHIFT_M = MODEL_X[0] - GRID_STEP_M*_MODEL_INDICES[0]
_VOL = MODEL_DX_M*MODEL_DY_M*MODEL_DZ_M
_MULTIPLIER = G*_VOL*1e12
_2D = (FFT_N,FFT_N)
_KERNEL_AXIS = (np.arange(-KERNEL_RADIUS,KERNEL_RADIUS+1)*GRID_STEP_M - _SHIFT_M)


def kernel(depth_m):
    """Convolution kernel K[t-s+320] with exact 12.5 m stagger."""
    if depth_m <= 0:
        raise ValueError('Receiver must be separated from model cell centers')
    x2 = _KERNEL_AXIS**2
    r2 = x2[:,None]+x2[None,:]+depth_m**2
    return ((3*depth_m**2-r2)/(r2**2*np.sqrt(r2))).astype(np.float32)

@lru_cache(maxsize=12)
def kernel_fft(height_m):
    """Cache transfer functions for each allowed observation height."""
    h=float(height_m)
    if h<0 or not np.isfinite(h):raise ValueError('Invalid height')
    return np.stack([rfft2(kernel(z+h),s=_2D).astype(np.complex64)
                     for z in MODEL_Z],axis=0)


def _source_fft(density):
    sparse=np.zeros((SHAPE[0],321,321),np.float32)
    sparse[:,_MODEL_INDICES[:,None],_MODEL_INDICES[None,:]]=density
    return rfft2(sparse,s=_2D,axes=(-2,-1))


def forward(density, height_m=100.):
    density=np.asarray(density,np.float32)
    if density.shape!=SHAPE or not np.isfinite(density).all():
        raise ValueError(f'Expected finite density {SHAPE}')
    spectrum=np.sum(_source_fft(density)*kernel_fft(float(height_m)),axis=0)
    conv=irfft2(spectrum,s=_2D).real
    ij=KERNEL_RADIUS+_OBS_INDICES
    return (conv[np.ix_(ij,ij)]*_MULTIPLIER).astype(np.float32)


def migration(observed_gzz, height_m=100., depth_weight=True):
    """Exact discrete adjoint (with optional depth weight); same FFT kernel.

    Discrete migration = A^T data * (depth+height)^2 when depth_weight.
    The weight is part of the migration imaging definition, not the adjoint.
    """
    d=np.asarray(observed_gzz,np.float32)
    if d.shape!=OBS_SHAPE or not np.isfinite(d).all():
        raise ValueError(f'Expected finite Gzz {OBS_SHAPE}')
    sparse=np.zeros((321,321),np.float32)
    sparse[_OBS_INDICES[:,None],_OBS_INDICES[None,:]]=d
    f=rfft2(sparse,s=_2D)
    picked=KERNEL_RADIUS+_MODEL_INDICES
    out=np.empty(SHAPE,np.float32)
    for k,z in enumerate(MODEL_Z):
        # correlation with same kernel: FFT flipped kernel, not conj alone;
        # kernel includes x/y stagger, so explicit reverse-FFT is required.
        adj=irfft2(f * rfft2(kernel(z+height_m)[::-1,::-1],s=_2D),s=_2D)
        out[k]=(adj[np.ix_(picked,picked)]*_MULTIPLIER).astype(np.float32)
        if depth_weight:
            out[k]*=np.float32((z+height_m)**2)
    return out


def normalize_migration(volume):
    v=np.asarray(volume,np.float32)
    scale=max(float(np.max(np.abs(v))),1e-8)
    return v/scale,scale


def direct_forward(density, height_m=100.):
    """Slow exact direct operator for independent correctness testing (sparse models)."""
    rho=np.asarray(density,np.float64)
    iy,ix=np.meshgrid(MODEL_Y,MODEL_X,indexing='ij')
    oy,ox=np.meshgrid(OBS_Y,OBS_X,indexing='ij')
    out=np.zeros(OBS_SHAPE,np.float64)
    for k in range(24):
        js,is_=np.nonzero(rho[k]); weights=rho[k,js,is_]
        for j,i,w in zip(js,is_,weights):
            dz=MODEL_Z[k]+height_m
            rr=(ox-MODEL_X[i])**2+(oy-MODEL_Y[j])**2+dz**2
            out += _MULTIPLIER*w*(3*dz**2-rr)/(rr**2*np.sqrt(rr))
    return out.astype(np.float32)
