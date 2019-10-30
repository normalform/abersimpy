"""
muscle.py
"""
import numpy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

from consts import WaveSpeedParamId, MassDensityParamId, NonLinearityCoefficientParamId, ConstantOfAttenuationParamId, \
    ExponentOfAttenuationParamId
from material.material import BaseMaterial


class Muscle(BaseMaterial):
    """
    Muscle
    """

    def __init__(self,
                 temperature: float):
        super().__init__(temperature)

    @property
    def constant_of_attenuation(self) -> float:
        """
        Get the constant of the attenuation
        :param: the temperature
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
        :param: the temperature
        :return: The exponent of the attenuation
        """
        # This is not correct!
        temperatures = numpy.array([20.0, 40.0])
        measurements = numpy.array([1.1, 1.1])

        return self._interpolation(temperatures, measurements)

    @property
    def non_linearity_coefficient(self) -> float:
        """
        Get the exponent of the attenuation
        :param: the temperature
        :return: The exponent of the attenuation
        """
        temperatures = numpy.array([36.0, 37.0])
        measurements = 1.0 + 0.5 * numpy.array([5.8, 5.8])
        return self._interpolation(temperatures, measurements)

    @property
    def mass_density(self) -> float:
        """
        Get the density
        :param: the temperature
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
    def wave_speed(self) -> float:
        """
        Get the wave speed
        :param: the temperature
        :return: wave speed
        """
        # 18 celsius, cardiac
        c0 = 1529.0
        scale = 1.1
        temperatures = numpy.array([18.0, 40.0])
        measurements = numpy.array([c0, c0 + scale * (40.0 - 18.0)])

        return self._interpolation(temperatures, measurements)


def muscle(param, temp):
    value = 0.0
    temps = []
    meas = []

    if param == WaveSpeedParamId:
        # 18 celsius, cardiac
        c0 = 1529.0
        scale = 1.1
        temps = numpy.array([18, 40])
        meas = numpy.array([c0, c0 + scale * (40 - 18)])
    elif param == MassDensityParamId:
        # 37 celsius
        rho0 = 1060.0
        temp0 = 37.0
        # volume expansion cow
        alpha0 = 3.75e-4
        temps = numpy.arange(35, 41)
        meas = rho0 * numpy.exp(alpha0 * (temp0 - temps))
    elif param == NonLinearityCoefficientParamId:
        temps = numpy.array([36, 37])
        meas = 1 + 0.5 * numpy.array([5.8, 5.8])
    elif param == ConstantOfAttenuationParamId:
        temps = numpy.array([36, 37])
        # db / cm
        meas = numpy.array([.52, .52])
    elif param == ExponentOfAttenuationParamId:
        # This is not correct!
        temps = numpy.array([20, 40])
        meas = numpy.array([1.1, 1.1])

    if len(meas) > 2:
        f = InterpolatedUnivariateSpline(temps, meas, ext='extrapolate')
        value = f(temp)
    else:
        f = interp1d(temps, meas, fill_value='extrapolate')
        value = f(temp)

    return value.item()