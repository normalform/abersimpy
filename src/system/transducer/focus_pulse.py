# -*- coding: utf-8 -*-
"""
    focus_pulse.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
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
    focus_azimuth = control.transducer.focus_azimuth
    focus_elevation = control.transducer.focus_elevation
    num_elements_azimuth = control.transducer.num_elements_azimuth
    num_elements_elevation = control.transducer.num_elements_elevation
    elements_size_azimuth = control.transducer.elements_size_azimuth
    elements_size_elevation = control.transducer.elements_size_elevation
    resolution_x = control.signal.resolution_x
    resolution_y = control.signal.resolution_y
    resolution_t = control.signal.resolution_t
    diffraction_type = control.diffraction_type
    annular_transducer = control.annular_transducer
    sound_speed = control.material.material.sound_speed

    if isinstance(physical_lens, int) is False:
        raise NotImplementedError
    physical_lens_x = physical_lens
    physical_lens_y = physical_lens

    # find sizes and indices
    surface_indexes_x, surface_indexes_y, _, _, _ = get_transducer_indexes(control)
    num_points_t, num_wave_field_y, num_wave_field_x = wave_field.shape
    if num_wave_field_x == 1:
        num_wave_field_x = num_wave_field_y
        num_wave_field_y = 1
    if num_wave_field_x >= numpy.max(surface_indexes_x) \
            and num_wave_field_y >= numpy.max(surface_indexes_y):
        raise NotImplementedError
    else:
        surface_indexes_x = range(num_wave_field_x)
        surface_indexes_y = range(0, num_wave_field_y)
        focused_wave_field = wave_field
    num_surface_x = len(surface_indexes_x)
    num_surface_y = len(surface_indexes_y)

    # calculate focusing
    if annular_transducer:
        raise NotImplementedError
    else:
        # straight forward rectangular transducer
        focal_curvature_x = get_focal_curvature(focus_azimuth,
                                                num_surface_x,
                                                num_elements_azimuth,
                                                resolution_x,
                                                elements_size_azimuth,
                                                _lens_focusing[0],
                                                annular_transducer,
                                                diffraction_type)
        focal_curvature_y = get_focal_curvature(focus_elevation,
                                                num_surface_y,
                                                num_elements_elevation,
                                                resolution_y,
                                                elements_size_elevation,
                                                _lens_focusing[1],
                                                annular_transducer,
                                                diffraction_type)
        if isinstance(focal_curvature_y, numpy.ndarray) is False:
            focal_curvature_y = numpy.array([focal_curvature_y])
        delta_focus_x = numpy.ones((num_surface_y, 1)) * focal_curvature_x.T / sound_speed
        delta_focus_y = focal_curvature_y[..., numpy.newaxis] / sound_speed * numpy.ones(
            num_surface_x)
        if physical_lens_x != 0:
            delta_focus_x = delta_focus_x - numpy.max(delta_focus_x)
        else:
            delta_focus_x = delta_focus_x - numpy.min(delta_focus_x)
        if physical_lens_y != 0:
            delta_focus_y = delta_focus_y - numpy.max(delta_focus_y)
        else:
            delta_focus_y = delta_focus_y - numpy.min(delta_focus_y)
        delta_focus = delta_focus_x + delta_focus_y

    _wave_field = wave_field
    if no_focusing is False:
        _wave_field[:, slice(surface_indexes_y[0], surface_indexes_y[-1] + 1),
        slice(surface_indexes_x[0], surface_indexes_x[-1] + 1)] = \
            _time_shift(focused_wave_field, -delta_focus / resolution_t, 'fft')

    return _wave_field, delta_focus


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
    num_dimensions = signal.ndim
    num_points_t, num_points_y, num_points_x = signal.shape

    if num_dimensions == 2:
        num_samples = num_points_y
        _signal = signal
    else:
        num_samples = num_points_x * num_points_y
        _signal = signal.reshape((num_points_t, num_samples))

    _delta = delta.reshape((1, delta.size))
    if numpy.max(_delta.shape) != num_samples and numpy.max(_delta.shape) > 1:
        raise NotImplementedError

    if method == 'fft':
        _signal = numpy.fft.fftn(_signal, axes=(0,))
        k = get_frequencies(num_points_t, 1)[..., numpy.newaxis]
        if numpy.max(_delta.shape) == 1:
            _delta = numpy.ones(num_samples) * _delta
        sh = numpy.exp(-1j * 2 * numpy.pi * k * _delta)
        shifted_signal = _signal * sh
        shifted_signal = numpy.fft.ifftn(shifted_signal, axes=(0,)).real
    else:
        raise NotImplementedError

    shifted_signal = shifted_signal.reshape(num_points_t, num_points_y, num_points_x)
    return shifted_signal
