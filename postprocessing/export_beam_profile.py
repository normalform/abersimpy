"""
export_beam_profile.py
"""
import numpy
from scipy.signal import hilbert

from controls.consts import NoHistory, PositionHistory, ProfileHistory, FullHistory, PlaneHistory, \
    PlaneByChannelHistory
from controls.main_control import MainControl
from filter.bandpass import bandpass
from misc.get_string_position import get_string_position


def export_beam_profile(control: MainControl,
                        wave_field: numpy.ndarray,
                        rms_profile: numpy.ndarray = None,
                        max_profile: numpy.ndarray = None,
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
            The index of the harm-dimension gives the n'th _harmonic profile starting
            with 0 as the total field.
        Maximum pressure in the temporal direction. The profile is structured as the RMS profile.
            The maximum is found as the maximum of the envelope of
            the signal at each point in space.
        Signal located at the control.transducer._center_channel index in space.
            This is the raw signal without any filtering performed for each step.
        The z-coordinate of each profile and axial pulse.
    """
    # setting variables
    _file_name = control.simulation_name

    _history = control.history
    _position = control.simulation.current_position
    _file_name = '{}{}.json'.format(_file_name, get_string_position(_position * 1e3))

    # stores full field or exits
    if _history == NoHistory:
        # No _history --> Exit
        raise NotImplementedError
    elif _history == FullHistory:
        # Saving pulse for each step, then exit
        print('save pulse to {}'.format(_file_name))
        raise NotImplementedError

    _harmonic = control.harmonic
    _store_position = control.simulation.store_position
    _center_channel = control.transducer.center_channel.astype(int)

    _transmit_frequency = control.signal.transmit_frequency
    _resolution_t = control.signal.resolution_t
    _filter = control.signal.filter

    # initializing profiles
    _rms_profile = rms_profile
    _max_profile = max_profile
    _ax_pulse = ax_pulse
    _z_coordinate = z_coordinate

    # finding dimensions
    _num_dimensions = control.num_dimensions
    if _num_dimensions == 2:
        _num_points_t, _num_points_y = wave_field.shape
        _num_points_x = _num_points_y
        _num_points_y = 1
    else:
        _num_points_t, _num_points_y, _num_points_x = wave_field.shape
    if step is None:
        _num_periods = 0
        _z_coordinate = 0
    else:
        _num_periods = step

    # saving pulse for steps specified in _store_position
    if _store_position.size != 0:
        if numpy.min(numpy.abs(_store_position - _position)) < 1e-12:
            print('[DUMMY] save pulse to {}'.format(_file_name))

    if _history is PositionHistory:
        return _rms_profile, _max_profile, _ax_pulse, _z_coordinate
    elif _history is PlaneByChannelHistory:
        raise NotImplementedError
    elif _history is ProfileHistory:
        if _num_dimensions == 2:
            _ax_pulse[:, _num_periods] = wave_field[:, _center_channel[0, ...]]
        else:
            _ax_pulse[:, _num_periods] = wave_field[:, _center_channel[1], _center_channel[0, ...]]

        if _num_periods == 0:
            _rms_profile = numpy.zeros_like(_rms_profile)
            _max_profile = numpy.zeros_like(_max_profile)

        _wave_field = wave_field.reshape((_num_points_t, _num_points_x * _num_points_y))
        _rms = _get_rms(_wave_field)
        _max = _get_max(_wave_field)
        _rms_profile[..., _num_periods, 0] = _rms.reshape((_num_points_y, _num_points_x))
        _max_profile[..., _num_periods, 0] = _max.reshape((_num_points_y, _num_points_x))
        _z_coordinate[_num_periods] = _position

        # filtering out harmonics
        for _harmonic_index in range(_harmonic):
            _index = _harmonic_index + 1
            _wave_field = wave_field.reshape((_num_points_t, _num_points_x * _num_points_y))
            _wave_field, _ = bandpass(_wave_field,
                                      numpy.array([_index * _transmit_frequency]),
                                      _resolution_t,
                                      _index * _transmit_frequency * _filter[_harmonic_index],
                                      4)
            _rms = _get_rms(_wave_field)
            _max = _get_max(_wave_field)
            _rms_profile[..., _num_periods, _harmonic_index + 1] = _rms.reshape((_num_points_y,
                                                                                 _num_points_x))
            _max_profile[..., _num_periods, _harmonic_index + 1] = _max.reshape((_num_points_y,
                                                                                 _num_points_x))
    elif _history is PlaneHistory:
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
        _maximum_pressure_profile = numpy.max(numpy.abs(hilbert(wave_field, axis=0)), axis=0)
    else:
        _maximum_pressure_profile = numpy.max(wave_field, axis=0)

    return _maximum_pressure_profile


def _get_rms(wave_field,
             scale_flag: bool = True):
    """
    Returns the RMS pressure profile of a signal.
    :param wave_field: The wave field.
    :param scale_flag: Flag specifying to use the RMS (sqrt(1/N*sum(u^2))
        or root of SOS (sqrt(sum(u^2)).
    :return: The RMS pressure profile of a signal
    """
    if scale_flag:
        _n = wave_field.shape[0]
        _rms_pressure_profile = numpy.sqrt(1 / _n * numpy.sum(wave_field ** 2, axis=0))
    else:
        _rms_pressure_profile = numpy.sqrt(numpy.sum(wave_field ** 2, axis=0))

    return _rms_pressure_profile