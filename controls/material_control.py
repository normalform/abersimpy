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
        self._thickness = self._get_thickness(material)
        self._offset = self._get_offset()

        _num_screens = 8
        self._num_screens = _num_screens
        self._delay_screens_amplitude = self._calc_delay_screens_amplitude(_num_screens)
        self._delay_screens_length = self._calc_delay_screens_length(_num_screens)
        self._delay_screens_seed = self._calc_delay_screens_seed(_num_screens)
        self._delay_screens_file = self._get_num_screen_filename(heterogeneous_medium)

    @staticmethod
    def _get_thickness(material: IMaterial) -> float:
        if isinstance(material, AberrationPhantom):
            _thickness = 0.035
        else:
            _thickness = 0.02

        return _thickness

    @staticmethod
    def _get_offset() -> List[float]:
        return [0.0, 0.0]

    @staticmethod
    def _calc_delay_screens_amplitude(num_screens: int) -> numpy.ndarray:
        return 0.09 * numpy.ones((num_screens, 1)) * 1e-3

    @staticmethod
    def _calc_delay_screens_length(num_screens: int) -> numpy.ndarray:
        return numpy.ones((num_screens, 1)) * 1e-3 * numpy.array([4, 100])

    @staticmethod
    def _calc_delay_screens_seed(num_screens: int) -> numpy.ndarray:
        return numpy.arange(1, num_screens + 1)

    @staticmethod
    def _get_num_screen_filename(heterogeneous_medium: int) -> str:
        if heterogeneous_medium == consts.AberrationFromFile:
            _delay_screens_file = 'randseq.json'
        elif heterogeneous_medium == consts.AberrationPhantom:
            _delay_screens_file = 'phantom.json'
        else:
            _delay_screens_file = ''

        return _delay_screens_file

    @property
    def material(self) -> IMaterial:
        return self._material

    @property
    def thickness(self) -> float:
        return self._thickness

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
