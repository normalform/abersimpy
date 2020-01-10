# -*- coding: utf-8 -*-
"""
    attenuation_solve.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import numpy
from scipy.signal import hilbert

from simulation.filter.get_frequencies import get_frequencies


def attenuation_solve(sample_points,
                      pulse,
                      resolution_z,
                      eps_a,
                      eps_b):
    """
    Imposing frequency dependent attenuation on the wave field.
    :param sample_points: Sampling points in time of the pulse.
    :param pulse: The pulse given at sampling times.
    :param resolution_z: Spatial step.
    :param eps_a: Attenuation constant.
    :param eps_b: Attenuation exponent
    :return: The propagated pulse at sampling times.
    """
    resolution_t = sample_points[1] - sample_points[0]
    _pulse = numpy.array(pulse)
    num_points_t = numpy.max(_pulse.shape)
    loss = 2 * numpy.pi * get_frequencies(num_points_t, resolution_t)

    # prepare attenuation coefficients
    loss = eps_a * numpy.conj(hilbert(numpy.abs(loss) ** eps_b)) * resolution_z
    loss = numpy.exp(-loss)

    pulse_f = numpy.fft.fftn(_pulse, axes=(0,))
    propagated_pulse_f = loss * pulse_f
    propagated_pulse = numpy.fft.ifftn(propagated_pulse_f, axes=(0,)).real

    return propagated_pulse
