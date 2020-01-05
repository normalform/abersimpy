"""
aberration_phantom.py

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

from system.material.material import BaseMaterial


class AberrationPhantom(BaseMaterial):
    """
    AberrationPhantom
    TODO Need unit tests
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
