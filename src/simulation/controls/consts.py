"""
const.py

Copyright (C) 2020  Jaeho Kim

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
