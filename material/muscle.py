from consts import WaveSpeedParamId, MassDensityParamId, NonLinearityCoefficientParamId, ConstantOfAttenuationParamId, \
    ExponentOfAttenuationParamId

import numpy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d


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