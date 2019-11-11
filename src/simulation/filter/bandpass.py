"""
bandpass.py
"""
import math
from typing import Tuple

import numpy
from scipy.sparse import spdiags

from simulation.filter.get_frequencies import get_frequencies


def bandpass(signal: numpy.ndarray,
             center_frequency: numpy.ndarray,
             sampling_interval: float,
             bandwidth: float = 0.0,
             steepness: float = 4.0,
             attenuation: float = -6.0) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Performs a bandpass filtering of a signal with sampling frequency = 1 / sampling_interval.
    The filter which is used is
    exp( -alpha * (| f - center_frequency | / 0.5 * bandwidth) ^ steepness) where alpha is
    computed to give a -dB bandwidth.
    TODO Need unit tests
    TODO consider signal's data shape for fft operations
    :param signal: The signal to bandpass.
    :param center_frequency: The center frequency.
    :param sampling_interval: The sampling interval.
    :param bandwidth: The width of the bandpass filter (two sided).
    :param steepness: The steepness of the bandpass filter.
    :param attenuation: Attenuation(dB) of the bandpass filter.
    :return: The filtered signal and frequency components of filter.
    """
    _frequency_components_of_filter = numpy.array(0.0)
    if bandwidth == math.inf:
        return signal, _frequency_components_of_filter

    # find size of and reshape signal
    _num_points_t = signal.shape[0]
    _num_points_x = 1
    _num_points_y = 1

    num_dimensions = signal.ndim
    if num_dimensions == 2:
        _num_points_x = signal.shape[1]
        _signal = signal.reshape((_num_points_t, _num_points_x * _num_points_y))
    else:
        _signal = signal

    # performs filtering in the frequency domain
    _signal_in_frequency_domain = numpy.fft.fftn(_signal, axes=(0,))
    _frequencies = get_frequencies(_num_points_t, sampling_interval)

    _center_frequency0 = _find_center_frequency(center_frequency,
                                                _frequencies,
                                                _signal_in_frequency_domain)

    _bandwidth = _calculate_bandwidth(bandwidth, _center_frequency0)

    _output_signal = _filtering(attenuation,
                                _frequencies,
                                _center_frequency0,
                                _bandwidth,
                                steepness,
                                _num_points_t,
                                _signal_in_frequency_domain)

    if num_dimensions == 2:
        _output_signal = _output_signal.reshape((_num_points_t, _num_points_y, _num_points_x))

    return _output_signal, _frequency_components_of_filter


def _find_center_frequency(center_frequency, frequencies, signal_frequency_domain):
    if center_frequency.ndim == 2:
        _frequency_indices = numpy.where(
            center_frequency[0] <= frequencies <= center_frequency[1])[0]

        if _frequency_indices.size == 0:
            _center_frequency0 = numpy.mean(center_frequency)
        else:
            _temp_signal_frequency_domain = numpy.mean(
                numpy.abs(signal_frequency_domain[_frequency_indices, :]), axis=1)
            _max_frequency_index = numpy.argmax(_temp_signal_frequency_domain, axis=0)
            _center_frequency0 = numpy.mean(
                frequencies[_frequency_indices[_max_frequency_index]])
    else:
        _center_frequency0 = center_frequency[0]

    return _center_frequency0


def _calculate_bandwidth(bandwidth, center_frequency):
    if bandwidth == 0.0:
        _bandwidth = 0.5 * center_frequency
    elif bandwidth < 2.0:
        _bandwidth = bandwidth * center_frequency
    else:
        _bandwidth = bandwidth

    return _bandwidth


def _filtering(attenuation,
               frequencies,
               center_frequency,
               bandwidth,
               steepness,
               num_points,
               signal_in_frequency_domain):
    _alpha = numpy.log(10 ** (attenuation / 20.0))
    _frequency_components_of_filter = numpy.exp(_alpha * (numpy.abs(
        numpy.abs(frequencies) - center_frequency) / (bandwidth / 2.0)) ** steepness)
    _df = spdiags(_frequency_components_of_filter, 0, num_points, num_points)
    _signal_frequency_domain = _df * signal_in_frequency_domain
    _output_signal = numpy.fft.ifftn(_signal_frequency_domain, axes=(0,)).real

    return _output_signal
