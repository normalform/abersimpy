"""
time_shift.py
"""
import numpy

from filter.get_frequencies import get_frequencies


def time_shift(signal: numpy.ndarray,
               delta: numpy.ndarray,
               method='fft') -> numpy.ndarray:
    """
    Shift the elements in u by delta along the first dimension (cols).
    If delta is a scalar, then every column in u is shifted by the same amount.
    If delta is a vector, then u(i,:) is shifted by delta(i).
    Positive values of delta gives a shift down. Negative values a shift up.
    :param signal: Signal to be shifted.
    :param delta:  Number of samples to shift.
    :param method: Shifting method. May be set to any of the following:
        'wrap'  Provides a circular shift of round(delta)
        'zpad'  Provide a shift of round(delta) and zero-pads
        'fft'   Imposes delta as a phase shift using fft
    :return: The shifted signal
    """
    _num_dimensions = signal.ndim
    _num_points_t, _num_points_y, _num_points_x = signal.shape

    if _num_dimensions == 2:
        _num_samples = _num_points_y
        _signal = signal
    else:
        _num_samples = _num_points_x * _num_points_y
        _signal = signal.reshape((_num_points_t, _num_samples))

    _delta = delta.reshape((1, delta.size))
    if numpy.max(_delta.shape) != _num_samples and numpy.max(_delta.shape) > 1:
        raise NotImplementedError

    if method == 'fft':
        _signal = numpy.fft.fftn(_signal, axes=(0,))
        k = get_frequencies(_num_points_t, 1)[..., numpy.newaxis]
        if numpy.max(_delta.shape) == 1:
            _delta = numpy.ones(_num_samples) * _delta
        _sh = numpy.exp(-1j * 2 * numpy.pi * k * _delta)
        _shifted_signal = _signal * _sh
        _shifted_signal = numpy.fft.ifftn(_shifted_signal, axes=(0,)).real
    else:
        raise NotImplementedError

    _shifted_signal = _shifted_signal.reshape(_num_points_t, _num_points_y, _num_points_x)
    return _shifted_signal
