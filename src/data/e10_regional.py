"""Regional-scale E10 synthetic Gzz forward/adjoint migration and geometries.

Model: 24 x 160 x 160, 50 m horizontal, 25 m vertical.
Observation grid: 161 x 161, 50 m spacing, 100 m above model top.
Gzz in Eotvos; density in g/cm3, z positive downward.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

NZ, NY, NX = 24, 160, 160
DX, DY, DZ = 50.0, 50.0, 25.0
FLIGHT_HEIGHT = 100.0
G = 6.67e-11
SHAPE = (NZ, NY, NX)
OBS_SHAPE = (NY + 1, NX + 1)
GEOMETRIES = ('compact', 'multiple', 'dipping', 'salt_dome')
Y, X = np.meshgrid((np.arange(NY) + .5)*DY, (np.arange(NX) + .5)*DX, indexing='ij')
Z = (np.arange(NZ) + .5)*DZ
# Difference in grid indices i-j runs -159..160; kernel index 159 means i-j=0.
_t = np.arange(-159, 161, dtype=np.float64)
_offset = (_t - .5) * DX
_yy, _xx = np.meshgrid(_offset, _offset, indexing='ij')
_RH2 = _xx*_xx + _yy*_yy
_KERNELS = []
for depth in Z:
    dz = depth + FLIGHT_HEIGHT
    r2 = _RH2 + dz*dz
    _KERNELS.append(((3*dz*dz-r2)/(r2**2*np.sqrt(r2))).astype(np.float64))
# Density g/cm3 -> kg/m3, gravity SI -> Eotvos; volume in m3.
_FORWARD_MULT = G * DX * DY * DZ * 1000.0 * 1e9
# E08D-style depth weight (same depth^2 dependence and gamma convention).
_MIG_WEIGHT = ((Z + FLIGHT_HEIGHT)**2 / ((G * 1e8) * (np.sqrt(3*np.pi)/2)))


def forward_gzz(density):
    assert density.shape == SHAPE
    result = np.zeros(OBS_SHAPE, np.float64)
    for k in range(NZ):
        if np.any(density[k]):
            result += fftconvolve(density[k], _KERNELS[k], mode='full')[159:320,159:320] * _FORWARD_MULT
    return result.astype(np.float32)


def migration_gzz(observed):
    assert observed.shape == OBS_SHAPE
    result = np.empty(SHAPE, np.float32)
    for k in range(NZ):
        raw = fftconvolve(observed, _KERNELS[k][::-1,::-1], mode='full')[160:320,160:320]
        result[k] = raw * _MIG_WEIGHT[k]
    return result


def random_box(rng, large=False):
    """Rectangular body with broad independent ranges, possibly cropped by survey boundaries."""
    length = rng.uniform(200, 2400 if large else 1600)
    width = rng.uniform(150, 1200)
    thick = rng.uniform(75, 420)
    x0, y0 = rng.uniform(-200, 7900), rng.uniform(-200, 7900)
    z0 = rng.uniform(10, max(20, 600-thick))
    theta = rng.uniform(0, np.pi)
    u = (X-x0)*np.cos(theta)+(Y-y0)*np.sin(theta)
    v = -(X-x0)*np.sin(theta)+(Y-y0)*np.cos(theta)
    lateral = (abs(u)<length/2)&(abs(v)<width/2)
    z_mask = (Z >= z0)&(Z <= z0+thick)
    return z_mask[:,None,None] & lateral[None,:,:]


def make_density(geometry, rng):
    rho = np.zeros(SHAPE, np.float32)
    if geometry in ('compact','multiple'):
        count = 1 if geometry=='compact' else int(rng.integers(2,6))
        for _ in range(count):
            for _attempt in range(30):
                mask = random_box(rng, large=(geometry=='compact'))
                if np.count_nonzero(mask)>25:
                    break
            else:
                raise RuntimeError('Unable to place compact/multiple body')
            sign = rng.choice([-1.0,1.0]); value=sign*rng.uniform(.15,1.0)
            rho[mask] = value
    elif geometry=='dipping':
        theta=rng.uniform(0,np.pi)
        xc,yc=rng.uniform(0,8000,2)
        u=(X-xc)*np.cos(theta)+(Y-yc)*np.sin(theta)
        v=-(X-xc)*np.sin(theta)+(Y-yc)*np.cos(theta)
        length=rng.uniform(400,3200); width=rng.uniform(100,1100)
        dip=rng.uniform(-.35,.35)
        z0=rng.uniform(100,420)
        thickness=rng.uniform(60,230)
        mid=z0+dip*u
        mask=(abs(u)<length/2)&(abs(v)<width/2)
        shape=mask[None,:,:] & (np.abs(Z[:,None,None]-mid[None,:,:])<thickness/2)
        rho[shape]=rng.choice([-1.0,1.0])*rng.uniform(.15,1.0)
    elif geometry=='salt_dome':
        xc,yc=rng.uniform(300,7700,2)
        r2=(X-xc)**2+(Y-yc)**2
        stem_radius=rng.uniform(180,650)
        cap_radius=rng.uniform(stem_radius*1.25,min(stem_radius*2.5,1800))
        cap_z=rng.uniform(80,320)
        cap_thick=rng.uniform(80,220)
        stem=(r2[None,:,:]<stem_radius**2)&(Z[:,None,None]>=cap_z)
        cap=(r2[None,:,:]<cap_radius**2)&(abs(Z[:,None,None]-cap_z)<cap_thick/2)
        rho[stem|cap]=-rng.uniform(.15,.85)
    else:
        raise ValueError(geometry)
    if not np.any(rho):
        raise RuntimeError(f'Empty model for {geometry}')
    return rho
