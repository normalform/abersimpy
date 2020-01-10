# -*- coding: utf-8 -*-
"""
    pulse_generator.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
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

    transmit_frequency = control.signal.transmit_frequency
    amplitude = control.signal.amplitude
    bandwidth = control.signal.bandwidth
    num_periods = control.signal.num_periods
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t
    resolution_x = control.signal.resolution_x
    resolution_y = control.signal.resolution_y
    resolution_t = control.signal.resolution_t
    sound_speed = control.material.material.sound_speed
    annular_transducer = control.annular_transducer

    # set length of transducer
    index_x, index_y, _, _, _ = get_transducer_indexes(control)
    index_num_x = index_x.size
    index_num_y = index_y.size

    if _lens_focusing.size == 1:
        _lens_focusing = _lens_focusing * numpy.array([1, 1])

    # generate _signal
    if isinstance(signal, str):
        if str.lower(signal) == 'gaussian':
            t, tp, _signal, nrm = _gaussian(resolution_t,
                                            transmit_frequency,
                                            num_periods,
                                            num_points_t)
        elif str.lower(signal) == 'cosine':
            t = numpy.arange(-num_points_t / 2, num_points_t / 2) * resolution_t
            tp = num_periods / transmit_frequency
            ntp = int(numpy.round(tp / resolution_t))
            tp = numpy.arange(-numpy.ceil(ntp / 2),
                              numpy.floor(ntp / 2) + 1) * resolution_t
            sigp = numpy.cos(2.0 * numpy.pi * transmit_frequency * tp) * \
                   numpy.cos(numpy.pi * transmit_frequency / num_periods * tp)
            _signal = numpy.zeros(t.size)
            index = numpy.where(t < tp[0])[0]
            _signal[index[-1]: index[-1] + ntp + 1] = sigp
            _signal, _ = bandpass(_signal,
                                  transmit_frequency,
                                  resolution_t,
                                  bandwidth,
                                  4)
            nrm = numpy.max(numpy.abs(hilbert(_signal)))
        else:
            print('pulse not specified - uses gaussian')
            t, tp, _signal, nrm = _gaussian(resolution_t, transmit_frequency,
                                            num_periods, num_points_t)
    else:
        _signal = signal
        nrm = numpy.max(numpy.abs(hilbert(_signal)))

    _signal = _signal / nrm * amplitude

    # adjust length of _signal
    if _signal.size < num_points_t:
        ntzp = (num_points_t - _signal.size) / 2
        _signal = numpy.concatenate(
            (numpy.zeros(int((numpy.ceil(ntzp)))), _signal.reshape(_signal.size),
             numpy.zeros(int((numpy.floor(ntzp))))))
    elif _signal.size > 1.1 * num_points_t:
        print(
            'Increase your number of spatial grid points.\nMaximum truncation '
            'of _signal vector is 10 percent.')
        sys.exit(-1)
    else:
        nttr = int(numpy.ceil((_signal.size - num_points_t) / 2.0))
        _signal = _signal[nttr:num_points_t + nttr]

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
            _apodization = _get_apodization(index_num_x,
                                            index_num_y,
                                            'tukey',
                                            _apodization,
                                            annular_transducer)
        elif isinstance(apodization, int):
            _apodization = _get_apodization(index_num_x,
                                            index_num_y,
                                            'tukey',
                                            apodization,
                                            annular_transducer)
        else:
            raise NotImplementedError

        # create wave field
        xd_signal = _signal[..., numpy.newaxis] * _apodization.reshape(
            (index_num_x * index_num_y))
        xd_signal = xd_signal.reshape((num_points_t, index_num_y, index_num_x))
        xd_signal, delta_focus = focus_pulse(control,
                                             xd_signal,
                                             _lens_focusing,
                                             no_focusing_flag)

        # create full domain
        _signal = numpy.zeros((num_points_t, num_points_y, num_points_x))
        _signal[:, slice(index_y[0], index_y[-1] + 1), slice(index_x[0], index_x[-1] + 1)] = \
            xd_signal
    else:
        print('Source type is not implemented')
        sys.exit(-1)

    # squeeze y-direction for 2D sim
    _signal = numpy.squeeze(_signal)

    return _signal, delta_focus


def _gaussian(resolution_t, transmit_frequency, num_periods, num_points_t):
    t = numpy.arange(-num_points_t / 2, num_points_t / 2) * resolution_t
    tp = num_periods / transmit_frequency
    sig = numpy.sin(2.0 * numpy.pi * transmit_frequency * t) * numpy.exp(
        -(2.0 * t / tp) ** 2)
    nrm = 1.0

    return t, tp, sig, nrm


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
            apodization = tukeywin(nx,s)
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
            apodization = scipy.signal.tukey(num_points_x, cutoff_percentage)
        elif apodization_type == 'hamming':
            apodization = numpy.hamming(num_points_x)
        elif apodization_type == 'hanning':
            apodization = numpy.hanning(num_points_x)
        elif apodization_type == 'hann':
            apodization = scipy.signal.hann(num_points_x)
        elif apodization_type == 'rect':
            apodization = scipy.signal.boxcar(num_points_x)
        return apodization

    # calculate 2d and annular apodization
    if annular_transducer:
        if num_points_y == 1:
            # get apodization for axis-symmetric simulations
            apodization_x = _get_apodization(2 * num_points_x - 1,
                                             1,
                                             apodization_type,
                                             cutoff_percentage[0])
            apodization = apodization_x[num_points_x:]
        elif num_points_x == num_points_y:
            # get apodization for annual transducer in full 3D
            raise NotImplementedError
        else:
            # return rectangular apodization
            print('For annular transducers, num_points_x and num_points_y has to be the same')
            apodization = numpy.ones((num_points_y, num_points_x))
            return apodization
    else:
        if len(cutoff_percentage) == 2:
            # different windows in x and y
            apodization_x = _get_apodization(num_points_x,
                                             1,
                                             apodization_type,
                                             cutoff_percentage[0])
            apodization_y = _get_apodization(num_points_y,
                                             1,
                                             apodization_type,
                                             cutoff_percentage[1])
        else:
            # equal windows in x and y
            apodization_x = _get_apodization(num_points_x,
                                             1,
                                             apodization_type,
                                             cutoff_percentage[0])
            apodization_y = _get_apodization(num_points_y,
                                             1,
                                             apodization_type,
                                             cutoff_percentage[1])

        apodization = apodization_y[..., numpy.newaxis] * apodization_x.T[numpy.newaxis, ...]

    return apodization
