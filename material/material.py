"""
material.py
"""
import math
from abc import ABC, abstractmethod

import numpy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

from consts import ScaleForSpatialVariablesZ, ScaleForPressure, ScaleForTemporalVariable


class BaseMaterial(ABC):
    """
    Base class for Material
    """

    def __init__(self,
                 temperature: float):
        self._temperature = temperature

    @property
    @abstractmethod
    def constant_of_attenuation(self) -> float:
        """
        Get the constant of the attenuation
        :param: the temperature
        :return: The constant of the attenuation
        """

    @property
    @abstractmethod
    def exponent_of_attenuation(self) -> float:
        """
        Get the exponent of the attenuation
        :param: the temperature
        :return: The exponent of the attenuation
        """

    @property
    @abstractmethod
    def non_linearity_coefficient(self) -> float:
        """
        Get the exponent of the attenuation
        :param: the temperature
        :return: The exponent of the attenuation
        """

    @property
    @abstractmethod
    def mass_density(self) -> float:
        """
        Get the density
        :param: the temperature
        :return: The density
        """

    @property
    @abstractmethod
    def sound_speed(self) -> float:
        """
        Get the sound speed
        :param: the temperature
        :return: sound speed
        """

    @property
    def eps_a(self) -> float:
        """
        Get the coefficient eps_a of material
        :param: the temperature
        :return: The coefficient eps_a of material
        """
        const_att = self.constant_of_attenuation
        exp_att = self.exponent_of_attenuation

        return const_att / ((2.0 * math.pi) ** exp_att * 20.0 * math.log10(math.exp(1.0)))

    @property
    def eps_b(self) -> float:
        """
        Get the coefficient eps_b of material
        :param: the temperature
        :return: The coefficient eps_b of material
        """
        return self.exponent_of_attenuation

    @property
    def eps_n(self) -> float:
        """
        Get the coefficient eps_n of material
        p.72 in Ultralyd II.
        :param: the temperature
        :return: The coefficient eps_n of material
        """
        beta_n = self.non_linearity_coefficient
        kappa = self.compressibility
        sound_speed = self.sound_speed

        scale = ScaleForSpatialVariablesZ * ScaleForPressure / ScaleForTemporalVariable
        return beta_n * kappa * sound_speed * scale

    @property
    def compressibility(self) -> float:
        """
        Get the compressibility
        :param: the temperature
        :return: The compressibility
        """
        sound_speed = self.sound_speed
        rho = self.mass_density
        kappa = 1. / (sound_speed ** 2.0 * rho)

        return kappa

    @property
    def is_regular(self) -> bool:
        """
        Return True if it's regular
        :return: True if it's regular
        """
        return True

    def _interpolation(self,
                       temperatures: numpy.ndarray,
                       measures: numpy.ndarray) -> float:
        if len(measures) > 2:
            interp_func = InterpolatedUnivariateSpline(temperatures, measures, ext='extrapolate')
        else:
            interp_func = interp1d(temperatures, measures, fill_value='extrapolate')

        return interp_func(self._temperature).item()
