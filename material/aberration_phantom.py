"""
aberration_phantom.py
"""
import numpy

from material.material import BaseMaterial


class AberrationPhantom(BaseMaterial):
    """
    AberrationPhantom
    """

    def __init__(self,
                 temperature: float):
        super().__init__(temperature)

    @property
    def constant_of_attenuation(self) -> float:
        """
        Get the constant of the attenuation
        :return: The constant of the attenuation
        """
        temperatures = numpy.array([20.0, 30.0])
        measurements = numpy.array([0.5, 0.5])

        return self._interpolation(temperatures, measurements)

    @property
    def exponent_of_attenuation(self) -> float:
        """
        Get the exponent of the attenuation
        :return: The exponent of the attenuation
        """
        temperatures = numpy.array([20.0, 30.0])
        measurements = numpy.array([1.0, 1.0])

        return self._interpolation(temperatures, measurements)

    @property
    def non_linearity_coefficient(self) -> float:
        """
        Get the exponent of the attenuation
        :return: The exponent of the attenuation
        """
        temperatures = numpy.array([20.0, 30.0])
        measurements = numpy.array([0.0, 0.0])

        return self._interpolation(temperatures, measurements)

    @property
    def mass_density(self) -> float:
        """
        Get the density
        :return: The density
        """
        temperatures = numpy.array([20.0, 30.0])
        measurements = numpy.array([1250.0, 1250.0])

        return self._interpolation(temperatures, measurements)

    @property
    def sound_speed(self) -> float:
        """
        Get the wave speed
        :return: wave speed
        """
        temperatures = numpy.array([20.0, 30.0])
        measurements = numpy.array([1640.0, 1640.0])

        return self._interpolation(temperatures, measurements)
