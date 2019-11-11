"""
muscle.py
"""
import numpy

from system.material.material import BaseMaterial


class Muscle(BaseMaterial):
    """
    The Muscle material
    """

    def __init__(self, temperature: float):
        super().__init__(temperature)

    @property
    def constant_of_attenuation(self) -> float:
        """
        Get the constant of the attenuation
        :return: The constant of the attenuation
        """
        temperatures = numpy.array([36.0, 37.0])
        # db / cm
        measurements = numpy.array([0.52, 0.52])

        return self._interpolation(temperatures, measurements)

    @property
    def exponent_of_attenuation(self) -> float:
        """
        Get the exponent of the attenuation
        :return: The exponent of the attenuation
        """
        temperatures = numpy.array([20.0, 40.0])
        measurements = numpy.array([1.1, 1.1])

        return self._interpolation(temperatures, measurements)

    @property
    def non_linearity_coefficient(self) -> float:
        """
        Get the exponent of the attenuation
        :return: The exponent of the attenuation
        """
        temperatures = numpy.array([36.0, 37.0])
        measurements = 1.0 + 0.5 * numpy.array([5.8, 5.8])

        return self._interpolation(temperatures, measurements)

    @property
    def mass_density(self) -> float:
        """
        Get the density
        :return: The density
        """
        # 37 celsius
        rho0 = 1060.0
        temp0 = 37.0
        # volume expansion cow
        alpha0 = 3.75e-4
        temperatures = numpy.arange(35, 41)
        measurements = rho0 * numpy.exp(alpha0 * (temp0 - temperatures))

        return self._interpolation(temperatures, measurements)

    @property
    def sound_speed(self) -> float:
        """
        Get the wave speed
        :return: wave speed
        """
        # 18 celsius, cardiac
        sound_speed = 1529.0
        scale = 1.1
        temperatures = numpy.array([18.0, 40.0])
        measurements = numpy.array([sound_speed, sound_speed + scale * (40.0 - 18.0)])

        return self._interpolation(temperatures, measurements)
