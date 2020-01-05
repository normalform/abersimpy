"""
simulation_control.py

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
import numpy

from simulation.controls.domain_control import DomainControl
from simulation.controls.material_control import MaterialControl


class SimulationControl:
    """
    SimulationControl
    TODO Need unit tests
    """

    def __init__(self,
                 domain: DomainControl,
                 end_point: float,
                 focus_elevation: float,
                 focus_azimuth: float,
                 heterogeneous_medium: int,
                 material: MaterialControl):
        # simulation control parameters
        self._num_windows: int = 2
        self._shock_step: float = 0.5
        self._endpoint = end_point

        _store_position = [focus_elevation, focus_azimuth]

        if heterogeneous_medium:
            _store_position = numpy.array([_store_position, material.thickness])
        self._store_position = numpy.unique(_store_position)

        self._step_size = domain.step_size
        self._current_position: float = 0.0

    @property
    def step_size(self) -> float:
        return self._step_size

    @step_size.setter
    def step_size(self, value: float):
        self._step_size = value

    @property
    def current_position(self) -> float:
        return self._current_position

    @current_position.setter
    def current_position(self, value: float):
        self._current_position = value

    @property
    def num_windows(self) -> int:
        return self._num_windows

    @property
    def shock_step(self) -> float:
        return self._shock_step

    @property
    def endpoint(self) -> float:
        return self._endpoint

    @property
    def store_position(self) -> numpy.ndarray:
        return self._store_position
