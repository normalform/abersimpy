"""
material_control.py
"""
from typing import List

import numpy

from controls import consts
from material.aberration_phantom import AberrationPhantom
from material.interfaces import IMaterial


class MaterialControl:
    """
    MaterialControl
    TODO Need unit tests
    """

    def __init__(self,
                 material: IMaterial,
                 heterogeneous_medium: int):
        # material control parameters
        self._material = material
        if isinstance(material, AberrationPhantom):
            self._thickness = 0.035
        else:
            self._thickness = 0.02
        self._offset = [0.0, 0.0]
        _num_screens = 8
        self._num_screens = _num_screens
        self._delay_screens_amplitude = 0.09 * numpy.ones((_num_screens, 1)) * 1e-3
        self._delay_screens_length = numpy.ones((_num_screens, 1)) * 1e-3 * numpy.array([4, 100])
        self._delay_screens_seed = numpy.arange(1, _num_screens + 1)
        if heterogeneous_medium == consts.AberrationFromFile:
            self._delay_screens_file = 'randseq.mat'
        elif heterogeneous_medium == consts.AberrationPhantom:
            self._delay_screens_file = 'phantoml.mat'
        else:
            self._delay_screens_file = ''

    @property
    def material(self) -> IMaterial:
        return self._material

    @property
    def thickness(self) -> float:
        return self.thickness

    @property
    def offset(self) -> List[float]:
        return self._offset

    @property
    def num_screens(self) -> int:
        return self._num_screens

    @property
    def delay_screens_amplitude(self) -> numpy.ndarray:
        return self._delay_screens_amplitude

    @property
    def delay_screens_length(self) -> numpy.ndarray:
        return self._delay_screens_length

    @property
    def delay_screens_seed(self) -> numpy.ndarray:
        return self._delay_screens_seed

    @property
    def delay_screens_file(self) -> str:
        return self._delay_screens_file
