"""
pulse_generator.py

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
import sys

import numpy
import scipy.signal
from scipy.signal import hilbert

from simulation.controls.main_control import MainControl
from simulation.filter.bandpass import bandpass
from system.transducer.focus_pulse import focus_pulse
from system.transducer.get_transducer_indexes import get_transducer_indexes


def pulse_generator(control: MainControl,
                    source: str = 'transducer',
                    apodization: int = 0,
                    signal: str = 'gaussian',
                    lens_focusing: numpy.ndarray = None,
                    pos=[0, 0],
                    no_focusing_flag: bool = False):
    """
    Generates a pulse with a given frequency, given number of periods,
    and with the specified the resolution.
    :param control: Control for simulation
    :param source: Specified where the pulse should be emitted.
        'transducer': signal from transducer.
        'pointsource': signal from point source.
    :param apodization: Apodization. If apodization is given as an vector
        of length one or two, the apodization is found using a tukey window
        apodization. Apodization may also be specified as a vector
        of arbitrary length, which is interpolated to fit the transducer,
        or matrix consisting of weights for a fully arbitrary
        transducer shape. The last option is apodization as a string
        specifying the type of apodization wanted.
        for more help.
    :param signal: The transmitted signal from the transducer.
        'gaussian': apodization sinusoidal vibration with a Gaussian envelope.
            The frequency and bandwidth of the pulse are
            specified in the control.
        'cosine': apodization cosine vibration of num_periods periods and a half period
            cosine envelope. The frequency and num_periods are specified in
            the control. The pulse is filtered around transmit frequency with
            a Gaussian filter with bandwidth specified in the control.
    :param lens_focusing: Use a lens in addition to delay focusing.
        lens_focusing is a two element vector containing the focus radii in azimuth
        and elevation direction. If the radius is zero, no lens is assumed.
    :param pos: Vector containing x and y channel for placement of signal in 'pointsource'.
        The position is given in mm and denotes the deviation from the origin defined by
        the center channel index. The pos argument may be used to impose small
        shifts of the transducer for heterogeneous studies,
        but larger shifts should be imposed by control.offset.
    :param no_focusing_flag: Returns the wave field in an unfocused manner
        regardless of focal point settings in both control and lens_focusing.
        Default is False which implies focusing.
    :return:
    """
    if lens_focusing is None:
        _lens_focusing = numpy.zeros(2)
        if control.transducer.num_elements_azimuth == 1:
            _lens_focusing[0] = control.transducer.focus_azimuth
        if control.transducer.num_elements_elevation == 1:
            _lens_focusing[1] = control.transducer.focus_elevation
    else:
        _lens_focusing = lens_focusing

    _transmit_frequency = control.signal.transmit_frequency
    _amplitude = control.signal.amplitude
    _bandwidth = control.signal.bandwidth
    _num_periods = control.signal.num_periods
    _num_points_x = control.domain.num_points_x
    _num_points_y = control.domain.num_points_y
    _num_points_t = control.domain.num_points_t
    _resolution_x = control.signal.resolution_x
    _resolution_y = control.signal.resolution_y
    _resolution_t = control.signal.resolution_t
    _sound_speed = control.material.material.sound_speed
    _annular_transducer = control.annular_transducer

    # set length of transducer
    _index_x, _index_y, _, _, _ = get_transducer_indexes(control)
    _index_num_x = _index_x.size
    _index_num_y = _index_y.size

    if _lens_focusing.size == 1:
        _lens_focusing = _lens_focusing * numpy.array([1, 1])

    # generate _signal
    if isinstance(signal, str):
        if str.lower(signal) == 'gaussian':
            _t, _tp, _signal, _nrm = _gaussian(_resolution_t,
                                               _transmit_frequency,
                                               _num_periods,
                                               _num_points_t)
        elif str.lower(signal) == 'cosine':
            _t = numpy.arange(-_num_points_t / 2, _num_points_t / 2) * _resolution_t
            _tp = _num_periods / _transmit_frequency
            _ntp = int(numpy.round(_tp / _resolution_t))
            _tp = numpy.arange(-numpy.ceil(_ntp / 2),
                               numpy.floor(_ntp / 2) + 1) * _resolution_t
            _sigp = numpy.cos(2.0 * numpy.pi * _transmit_frequency * _tp) * \
                    numpy.cos(numpy.pi * _transmit_frequency / _num_periods * _tp)
            _signal = numpy.zeros(_t.size)
            _index = numpy.where(_t < _tp[0])[0]
            _signal[_index[-1]: _index[-1] + _ntp + 1] = _sigp
            _signal, _ = bandpass(_signal,
                                  _transmit_frequency,
                                  _resolution_t,
                                  _bandwidth,
                                  4)
            _nrm = numpy.max(numpy.abs(hilbert(_signal)))
        else:
            print('pulse not specified - uses gaussian')
            _t, _tp, _signal, _nrm = _gaussian(_resolution_t, _transmit_frequency,
                                               _num_periods, _num_points_t)
    else:
        _signal = signal
        _nrm = numpy.max(numpy.abs(hilbert(_signal)))

    _signal = _signal / _nrm * _amplitude

    # adjust length of _signal
    if _signal.size < _num_points_t:
        _ntzp = (_num_points_t - _signal.size) / 2
        _signal = numpy.concatenate(
            (numpy.zeros(int((numpy.ceil(_ntzp)))), _signal.reshape(_signal.size),
             numpy.zeros(int((numpy.floor(_ntzp))))))
    elif _signal.size > 1.1 * _num_points_t:
        print(
            'Increase your number of spatial grid points.\nMaximum truncation '
            'of _signal vector is 10 percent.')
        sys.exit(-1)
    else:
        _nttr = int(numpy.ceil((_signal.size - _num_points_t) / 2.0))
        _signal = _signal[_nttr:_num_points_t + _nttr]

    if control.num_dimensions == 1:
        return _signal

    # choose source
    if source == 'pointsource':
        raise NotImplementedError
    elif source == 'impulse':
        raise NotImplementedError
    elif source == 'transducer':
        # generating pulse from transducer
        control.simulation.current_position = 0.0

        # calculate apodization
        if isinstance(apodization, str):
            raise NotImplementedError
        elif isinstance(apodization, list):
            _apodization = numpy.array(apodization)
            _apodization = _get_apodization(_index_num_x,
                                            _index_num_y,
                                            'tukey',
                                            _apodization,
                                            _annular_transducer)
        elif isinstance(apodization, int):
            _apodization = _get_apodization(_index_num_x,
                                            _index_num_y,
                                            'tukey',
                                            apodization,
                                            _annular_transducer)
        else:
            raise NotImplementedError

        # create wave field
        _xd_signal = _signal[..., numpy.newaxis] * _apodization.reshape(
            (_index_num_x * _index_num_y))
        _xd_signal = _xd_signal.reshape((_num_points_t, _index_num_y, _index_num_x))
        _xd_signal, _delta_focus = focus_pulse(control,
                                               _xd_signal,
                                               _lens_focusing,
                                               no_focusing_flag)

        # create full domain
        _signal = numpy.zeros((_num_points_t, _num_points_y, _num_points_x))
        _signal[:, slice(_index_y[0], _index_y[-1] + 1), slice(_index_x[0], _index_x[-1] + 1)] = \
            _xd_signal
    else:
        print('Source type is not implemented')
        sys.exit(-1)

    # squeeze y-direction for 2D sim
    _signal = numpy.squeeze(_signal)

    return _signal, _delta_focus


def _gaussian(resolution_t, transmit_frequency, num_periods, num_points_t):
    _t = numpy.arange(-num_points_t / 2, num_points_t / 2) * resolution_t
    _tp = num_periods / transmit_frequency
    _sig = numpy.sin(2.0 * numpy.pi * transmit_frequency * _t) * numpy.exp(
        -(2.0 * _t / _tp) ** 2)
    _nrm = 1.0

    return _t, _tp, _sig, _nrm


def _get_apodization(num_points_x: int,
                     num_points_y: int = 1,
                     apodization_type: str = 'tukey',
                     cutoff_percentage=0,
                     annular_transducer=False):
    """
    Returns apodization window for a transducer of length nx.
    :param num_points_x: Number of points in x-direction.
    :param num_points_y: Number of points in y-direction. Default is 1.
    :param apodization_type: Specifies type of apodization. May be either
        'rect', 'tukey', 'hamming','hanning' or 'hann' (Hanning-window with zero endpoints).
        The default is 'tukey'.
    :param cutoff_percentage: Specifies the cut-off percentage for a tukey-window.
        First element is apodization in the x-direction and the second in the y-direction.
        For a scalar, the same value is used for both x and y. Default value is 1.
            _apodization = tukeywin(nx,s)
            For s=1 tueky is a hann window and s=0 it is a rect.
            Note that the apodization is rectangular for s=0, and will go towards a rectangular
            apodization with zero at the endpoints for s -> Inf.
            Typical values for cosine apodization are in the range of 1-10.
    :param annular_transducer: Specifies an annular transducer. Will for a 3D
        simulation return a 2D annular window. For an axi-symmetric simulation
        the apodization window is a 1D vector. Default is annular_transducer = False,
        which implies a rectangular transducer.
    :return: Normalized apodization window.
    """
    # return apodization based on apodization_type
    if num_points_y == 1 and annular_transducer is False and isinstance(cutoff_percentage,
                                                                        list) is False:
        if apodization_type == 'tukey':
            _apodization = scipy.signal.tukey(num_points_x, cutoff_percentage)
        elif apodization_type == 'hamming':
            _apodization = numpy.hamming(num_points_x)
        elif apodization_type == 'hanning':
            _apodization = numpy.hanning(num_points_x)
        elif apodization_type == 'hann':
            _apodization = scipy.signal.hann(num_points_x)
        elif apodization_type == 'rect':
            _apodization = scipy.signal.boxcar(num_points_x)
        return _apodization

    # calculate 2d and annular apodization
    if annular_transducer:
        if num_points_y == 1:
            # get apodization for axis-symmetric simulations
            _apodization_x = _get_apodization(2 * num_points_x - 1,
                                              1,
                                              apodization_type,
                                              cutoff_percentage[0])
            _apodization = _apodization_x[num_points_x:]
        elif num_points_x == num_points_y:
            # get apodization for annual transducer in full 3D
            raise NotImplementedError
        else:
            # return rectangular apodization
            print('For annular transducers, num_points_x and num_points_y has to be the same')
            _apodization = numpy.ones((num_points_y, num_points_x))
            return _apodization
    else:
        if len(cutoff_percentage) == 2:
            # different windows in x and y
            _apodization_x = _get_apodization(num_points_x,
                                              1,
                                              apodization_type,
                                              cutoff_percentage[0])
            _apodization_y = _get_apodization(num_points_y,
                                              1,
                                              apodization_type,
                                              cutoff_percentage[1])
        else:
            # equal windows in x and y
            _apodization_x = _get_apodization(num_points_x,
                                              1,
                                              apodization_type,
                                              cutoff_percentage[0])
            _apodization_y = _get_apodization(num_points_y,
                                              1,
                                              apodization_type,
                                              cutoff_percentage[1])

        _apodization = _apodization_y[..., numpy.newaxis] * _apodization_x.T[numpy.newaxis, ...]

    return _apodization
