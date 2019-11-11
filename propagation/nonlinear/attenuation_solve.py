"""
attenuation_solve.py
"""
import numpy
from scipy.signal import hilbert

from filter.get_frequencies import get_frequencies


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
    _resolution_t = sample_points[1] - sample_points[0]
    _pulse = numpy.array(pulse)
    _num_points_t = numpy.max(_pulse.shape)
    _loss = 2 * numpy.pi * get_frequencies(_num_points_t, _resolution_t)

    # prepare attenuation coefficients
    _loss = eps_a * numpy.conj(hilbert(numpy.abs(_loss) ** eps_b)) * resolution_z
    _loss = numpy.exp(-_loss)

    _pulse_f = numpy.fft.fftn(_pulse, axes=(0,))
    _propagated_pulse_f = _loss * _pulse_f
    _propagated_pulse = numpy.fft.ifftn(_propagated_pulse_f, axes=(0,)).real

    return _propagated_pulse
