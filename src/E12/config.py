"""Independent physical geometry and experiment definitions for E12.

81 receivers span 8 km (100 m intervals). 64 density cells span the
same 8 km (125 m cells). The receiver and density grids are *not* coincident.
"""
from dataclasses import dataclass

SHAPE = (24,64,64)
OBS_SHAPE = (81,81)
MODEL_DX_M = MODEL_DY_M = 125.0
MODEL_DZ_M = 1000.0/24
OBS_DX_M = OBS_DY_M = 100.0
EXTENT_M = 8000.0
FLIGHT_HEIGHT_M = 100.0  # nominal survey flight height from Jorgensen & Zhdanov 2021, §6.2
GRID_STEP_M = 25.0  # shared lattice: receiver pitch=4, model cell pitch=5
FFT_N = 1024
KERNEL_RADIUS = 320
GEOMETRIES_4 = ('compact','multiple','dipping','salt_dome')
GEOMETRIES_5 = GEOMETRIES_4 + ('broad_intrusion',)
HEIGHTS_VARIABLE = (0.,50.,100.,150.,200.)
NOISES = (0.,5.,10.,15.)

@dataclass(frozen=True)
class Experiment:
    dataset: str
    geometries: tuple
    heights: tuple
    physics_weight: float = 0.0
    tversky_weight: float = 0.1
    depth_weight: float = 0.02

EXPERIMENTS = {
    'E12BASE': Experiment('D12_07', GEOMETRIES_4, (0.,)),
    'E12A': Experiment('D12_08', GEOMETRIES_5, (0.,)),
    'E12B': Experiment('D12_09', GEOMETRIES_4, HEIGHTS_VARIABLE),
    'E12C': Experiment('D12_07', GEOMETRIES_4, (0.,), .01),
    'E12D': Experiment('D12_10', GEOMETRIES_5, HEIGHTS_VARIABLE),
    'E12E': Experiment('D12_10', GEOMETRIES_5, HEIGHTS_VARIABLE, .01),
    'E12F': Experiment('D12_10', GEOMETRIES_5, HEIGHTS_VARIABLE, .01, .02, .005),
}
DATASETS = {
    'D12_07': (GEOMETRIES_4, (0.,)),
    'D12_08': (GEOMETRIES_5, (0.,)),
    'D12_09': (GEOMETRIES_4, HEIGHTS_VARIABLE),
    'D12_10': (GEOMETRIES_5, HEIGHTS_VARIABLE),
}
