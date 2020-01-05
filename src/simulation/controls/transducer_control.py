"""
transducer_control.py

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
from system.diffraction.diffraction import PseudoDifferential, \
    FiniteDifferenceTimeDifferenceFull, FiniteDifferenceTimeDifferenceReduced
from system.diffraction.interfaces import IDiffractionType


class TransducerControl:
    """
    TransducerControl
    TODO Need unit tests
    """

    def __init__(self,
                 domain: DomainControl,
                 num_elements_azimuth: int,
                 num_elements_elevation: int,
                 elements_size_azimuth: float,
                 elements_size_elevation: float,
                 focus_azimuth: float,
                 focus_elevation: float,
                 diffraction_type: IDiffractionType,
                 num_points_x: int,
                 num_points_y: int,
                 annular_transducer: bool):
        # transducer control parameters
        self._num_elements_azimuth = num_elements_azimuth
        self._num_elements_elevation = num_elements_elevation
        self._elements_size_azimuth = elements_size_azimuth
        self._elements_size_elevation = elements_size_elevation
        self._probe_span_azimuth = domain.probe_span_azimuth
        self._probe_span_elevation = domain.probe_span_elevation
        self._focus_azimuth = focus_azimuth
        self._focus_elevation = focus_elevation
        if annular_transducer and diffraction_type in (PseudoDifferential,
                                                       FiniteDifferenceTimeDifferenceReduced,
                                                       FiniteDifferenceTimeDifferenceFull):
            self._center_channel = numpy.array([1, 1])
        else:
            self._center_channel = numpy.floor([num_points_x / 2, num_points_y / 2]) + 1
        self._center_channel.astype(int)

    @property
    def num_elements_azimuth(self) -> int:
        return self._num_elements_azimuth

    @property
    def num_elements_elevation(self) -> int:
        return self._num_elements_elevation

    @property
    def elements_size_azimuth(self) -> float:
        return self._elements_size_azimuth

    @property
    def elements_size_elevation(self) -> float:
        return self._elements_size_elevation

    @property
    def probe_span_azimuth(self) -> float:
        return self._probe_span_azimuth

    @property
    def probe_span_elevation(self) -> float:
        return self._probe_span_elevation

    @property
    def focus_azimuth(self) -> float:
        return self._focus_azimuth

    @property
    def focus_elevation(self) -> float:
        return self._focus_elevation

    @property
    def center_channel(self) -> numpy.ndarray:
        return self._center_channel
