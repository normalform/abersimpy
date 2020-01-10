# -*- coding: utf-8 -*-
"""
    export_beam_profile.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
from typing import Optional

import numpy
from scipy.signal import hilbert

from simulation.controls.consts import NO_HISTORY, POSITION_HISTORY, \
    PROFILE_HISTORY, FULL_HISTORY, PLANE_HISTORY, PLANE_BY_CHANNEL_HISTORY
from simulation.controls.main_control import MainControl
from simulation.filter.bandpass import bandpass


def export_beam_profile(control: MainControl,
                        wave_field: numpy.ndarray,
                        rms_profile: Optional[numpy.ndarray] = None,
                        max_profile: Optional[numpy.ndarray] = None,
                        ax_pulse=None,
                        z_coordinate=None,
                        step=None):
    """
    exporting beam profiles.
    :param control: The controls.
    :param wave_field: The wave field to be exported.
    :param rms_profile: For export to an already existing beam profile
        of size (nx * ny * np * num_harm).
    :param max_profile: For export to an already existing beam profile.
    :param ax_pulse: For export to an already existing beam profile.
    :param z_coordinate: The z coordinate of existing profiles.
    :param step: The step number. Used to indicate profile index.
    :return:
        Temporal RMS beam profile for all frequencies. The profile has
            dimensions (ny * nx * np+1 * num_harm), possibly with ny as singleton dimension.
            The index of the harm-dimension gives the n'th harmonic profile starting
            with 0 as the total field.
        Maximum pressure in the temporal direction. The profile is structured as the RMS profile.
            The maximum is found as the maximum of the envelope of
            the signal at each point in space.
        Signal located at the control.transducer.center_channel index in space.
            This is the raw signal without any filtering performed for each step.
        The z-coordinate of each profile and axial pulse.
    """
    # setting variables
    file_name = control.simulation_name

    history = control.history
    position = control.simulation.current_position
    file_name = '{}{}.json'.format(file_name, _get_string_position(position * 1e3))

    # stores full field or exits
    if history == NO_HISTORY:
        # No history --> Exit
        raise NotImplementedError
    elif history == FULL_HISTORY:
        # Saving pulse for each step, then exit
        print('save pulse to {}'.format(file_name))
        raise NotImplementedError

    harmonic = control.harmonic
    store_position = control.simulation.store_position
    center_channel = control.transducer.center_channel.astype(int)

    transmit_frequency = control.signal.transmit_frequency
    resolution_t = control.signal.resolution_t
    filter = control.signal.filter

    # initializing profiles
    _rms_profile = rms_profile
    _max_profile = max_profile
    _ax_pulse = ax_pulse
    _z_coordinate = z_coordinate

    # finding dimensions
    num_dimensions = control.num_dimensions
    if num_dimensions == 2:
        num_points_t, num_points_y = wave_field.shape
        num_points_x = num_points_y
        num_points_y = 1
    else:
        num_points_t, num_points_y, num_points_x = wave_field.shape
    if step is None:
        num_periods = 0
        _z_coordinate = 0
    else:
        num_periods = step

    # saving pulse for steps specified in store_position
    if store_position.size != 0:
        if numpy.min(numpy.abs(store_position - position)) < 1e-12:
            print('[DUMMY] save pulse to {}'.format(file_name))

    if history is POSITION_HISTORY:
        return _rms_profile, _max_profile, _ax_pulse, _z_coordinate
    elif history is PLANE_BY_CHANNEL_HISTORY:
        raise NotImplementedError
    elif history is PROFILE_HISTORY:
        if num_dimensions == 2:
            _ax_pulse[:, num_periods] = wave_field[:, center_channel[0, ...]]
        else:
            _ax_pulse[:, num_periods] = wave_field[:, center_channel[1], center_channel[0, ...]]

        if num_periods == 0:
            _rms_profile = numpy.zeros_like(_rms_profile)
            _max_profile = numpy.zeros_like(_max_profile)

        wave_field = wave_field.reshape((num_points_t, num_points_x * num_points_y))
        _rms = _get_rms(wave_field)
        _max = _get_max(wave_field)
        _rms_profile[..., num_periods, 0] = _rms.reshape((num_points_y, num_points_x))
        _max_profile[..., num_periods, 0] = _max.reshape((num_points_y, num_points_x))
        _z_coordinate[num_periods] = position

        # filtering out harmonics
        for harmonic_index in range(harmonic):
            index = harmonic_index + 1
            wave_field = wave_field.reshape((num_points_t, num_points_x * num_points_y))
            wave_field, _ = bandpass(wave_field,
                                     numpy.array([index * transmit_frequency]),
                                     resolution_t,
                                     index * transmit_frequency * filter[harmonic_index],
                                     4)
            _rms = _get_rms(wave_field)
            _max = _get_max(wave_field)
            _rms_profile[..., num_periods, harmonic_index + 1] = _rms.reshape((num_points_y,
                                                                               num_points_x))
            _max_profile[..., num_periods, harmonic_index + 1] = _max.reshape((num_points_y,
                                                                               num_points_x))
    elif history is PLANE_HISTORY:
        raise NotImplementedError

    return _rms_profile, _max_profile, _ax_pulse, _z_coordinate


def _get_max(wave_field: numpy.ndarray,
             envelope_flag: bool = True) -> numpy.ndarray:
    """
    Returns the maximum pressure profile of a signal.
    :param wave_field: The wave field.
    :param envelope_flag: Flag specifying if the maximum of the envelope should be used.
    :return: The maximum pressure profile.
    """
    if envelope_flag != 0:
        maximum_pressure_profile = numpy.max(numpy.abs(hilbert(wave_field, axis=0)), axis=0)
    else:
        maximum_pressure_profile = numpy.max(wave_field, axis=0)

    return maximum_pressure_profile


def _get_rms(wave_field,
             scale_flag: bool = True):
    """
    Returns the RMS pressure profile of a signal.
    :param wave_field: The wave field.
    :param scale_flag: Flag specifying to use the RMS (sqrt(1/N*sum(pulse^2))
        or root of SOS (sqrt(sum(pulse^2)).
    :return: The RMS pressure profile of a signal
    """
    if scale_flag:
        n = wave_field.shape[0]
        rms_pressure_profile = numpy.sqrt(1 / n * numpy.sum(wave_field ** 2, axis=0))
    else:
        rms_pressure_profile = numpy.sqrt(numpy.sum(wave_field ** 2, axis=0))

    return rms_pressure_profile


def _get_string_position(position: float):
    """
    Function that returns a position as a string. Input is given in mm and
    the string containing 5 digits is organized such that the last digit
    represents a 1/100 of a millimeter. Any length up to 99.999 cm may be
    expressed uniquely up to a precision of 0.001 cm.
    :param position: The position.
    :return: Position as a string _XXXXX
    """
    _position = numpy.round(100 * position) / 100
    _position = _position / 100
    n = 6

    num = numpy.zeros(n, dtype=int)
    for index in range(n):
        num[index] = int(numpy.floor(_position))
        _position = _position - num[index]
        _position = 10 * _position

    temp = num[-1]
    cl = 0
    for index in range(n - 1):
        if num[n - index - 1] == 9 and temp == 9:
            cl = cl + 1
            temp = num[n - index - 1]
        else:
            temp = num[n - index - 1]

    if cl > 1:
        num[n - 1] = 0
        for index in range(cl):
            num[n - index - 1] = int(numpy.mod(num[n - index - 1] + 1, 10))

    output = '_{:1d}{:1d}{:1d}{:1d}{:1d}'.format(num[0], num[1], num[2], num[3], num[4])

    return output
