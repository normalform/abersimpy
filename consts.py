"""
const.py
"""
# Scale factors
ScaleForSpatialVariablesZ: float = 1e-2
ScaleForPressure: float = 1e6
ScaleForTemporalVariable: float = 1e-6

# Material parameter IDs
WaveSpeedParamId: int = 1
MassDensityParamId: int = 2
NonLinearityCoefficientParamId: int = 3
ConstantOfAttenuationParamId: int = 4
ExponentOfAttenuationParamId: int = 5

# History types
NoHistory: int = 0
PositionHistory: int = 1
ProfileHistory: int = 2
FullHistory: int = 3
PlaneHistory: int = 4
PlaneByChannelHistory: int = 5

# Aberration types
NoAberrationAndHomogeneousMedium: int = 0
AberrationFromDelayScreenBodyWall: int = 1
AberrationFromFile: int = 2
AberrationPhantom: int = 3
