"""
focus_pulse.py

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

from simulation.filter.get_frequencies import get_frequencies
from system.transducer.get_focal_curvature import get_focal_curvature
from system.transducer.get_transducer_indexes import get_transducer_indexes


def focus_pulse(control,
                wave_field,
                lens_focusing=None,
                no_focusing: bool = False,
                physical_lens=0):
    """
    Function that focuses a wave field according to the control.
    :param control: The control.
    :param wave_field: The wave field.
    :param lens_focusing: Additional lens focusing.
    :param no_focusing: If no_focusing is True, the field is not focused,
        but the focusing delays are calculated and returned.
    :param physical_lens: A flag specifying the use of a physical lens, that is a lens where
        the endpoints of the lens are un-altered, and the focusing delays are introduces
        for the interior points. An unphysical lens are adjusted such that the center channel
        does not move during focusing. If physical_lens has two numbers, they are associated
        with the size of the lens.
    :return: Focused wave field, Focal delays
    """
    if lens_focusing is None:
        _lens_focusing = numpy.zeros(2)
        if control.transducer.num_elements_azimuth == 1:
            _lens_focusing[0] = control.transducer.focus_azimuth
        if control.transducer.num_elements_elevation == 1:
            _lens_focusing[1] = control.transducer.focus_elevation
    else:
        _lens_focusing = lens_focusing

    # initiate variables
    _focus_azimuth = control.transducer.focus_azimuth
    _focus_elevation = control.transducer.focus_elevation
    _num_elements_azimuth = control.transducer.num_elements_azimuth
    _num_elements_elevation = control.transducer.num_elements_elevation
    _elements_size_azimuth = control.transducer.elements_size_azimuth
    _elements_size_elevation = control.transducer.elements_size_elevation
    _resolution_x = control.signal.resolution_x
    _resolution_y = control.signal.resolution_y
    _resolution_t = control.signal.resolution_t
    _diffraction_type = control.diffraction_type
    _annular_transducer = control.annular_transducer
    _sound_speed = control.material.material.sound_speed

    if isinstance(physical_lens, int) is False:
        raise NotImplementedError
    _physical_lens_x = physical_lens
    _physical_lens_y = physical_lens

    # find sizes and indices
    _surface_indexes_x, _surface_indexes_y, _, _, _ = get_transducer_indexes(control)
    _num_points_t, _num_wave_field_y, _num_wave_field_x = wave_field.shape
    if _num_wave_field_x == 1:
        _num_wave_field_x = _num_wave_field_y
        _num_wave_field_y = 1
    if _num_wave_field_x >= numpy.max(_surface_indexes_x) \
            and _num_wave_field_y >= numpy.max(_surface_indexes_y):
        raise NotImplementedError
    else:
        _surface_indexes_x = range(_num_wave_field_x)
        _surface_indexes_y = range(0, _num_wave_field_y)
        _focused_wave_field = wave_field
    _num_surface_x = len(_surface_indexes_x)
    _num_surface_y = len(_surface_indexes_y)

    # calculate focusing
    if _annular_transducer:
        raise NotImplementedError
    else:
        # straight forward rectangular transducer
        _focal_curvature_x = get_focal_curvature(_focus_azimuth,
                                                 _num_surface_x,
                                                 _num_elements_azimuth,
                                                 _resolution_x,
                                                 _elements_size_azimuth,
                                                 _lens_focusing[0],
                                                 _annular_transducer,
                                                 _diffraction_type)
        _focal_curvature_y = get_focal_curvature(_focus_elevation,
                                                 _num_surface_y,
                                                 _num_elements_elevation,
                                                 _resolution_y,
                                                 _elements_size_elevation,
                                                 _lens_focusing[1],
                                                 _annular_transducer,
                                                 _diffraction_type)
        if isinstance(_focal_curvature_y, numpy.ndarray) is False:
            _focal_curvature_y = numpy.array([_focal_curvature_y])
        _delta_focus_x = numpy.ones((_num_surface_y, 1)) * _focal_curvature_x.T / _sound_speed
        _delta_focus_y = _focal_curvature_y[..., numpy.newaxis] / _sound_speed * numpy.ones(
            _num_surface_x)
        if _physical_lens_x != 0:
            _delta_focus_x = _delta_focus_x - numpy.max(_delta_focus_x)
        else:
            _delta_focus_x = _delta_focus_x - numpy.min(_delta_focus_x)
        if _physical_lens_y != 0:
            _delta_focus_y = _delta_focus_y - numpy.max(_delta_focus_y)
        else:
            _delta_focus_y = _delta_focus_y - numpy.min(_delta_focus_y)
        _delta_focus = _delta_focus_x + _delta_focus_y

    _wave_field = wave_field
    if no_focusing is False:
        _wave_field[:, slice(_surface_indexes_y[0], _surface_indexes_y[-1] + 1),
        slice(_surface_indexes_x[0], _surface_indexes_x[-1] + 1)] = \
            _time_shift(_focused_wave_field, -_delta_focus / _resolution_t, 'fft')

    return _wave_field, _delta_focus


def _time_shift(signal: numpy.ndarray,
                delta: numpy.ndarray,
                method='fft') -> numpy.ndarray:
    """
    Shift the elements in pulse by delta along the first dimension (cols).
    If delta is a scalar, then every column in pulse is shifted by the same amount.
    If delta is a vector, then pulse(i,:) is shifted by delta(i).
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
