import numpy
from scipy.signal import hilbert

from filter.bandpass import bandpass
from transducer.focus_pulse import focus_pulse
from transducer.get_apodization import get_apodization
from transducer.get_xdidx import get_xdidx


def pulse_generator(control,
                    source: str = 'transducer',
                    apodization: int = 0,
                    signal: str = 'gaussian',
                    lensfoc=None,
                    pos=[0, 0],
                    nofocflag=0):
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
        specifying the type of apodization wanted. see get_apodization'
        for more help.
    :param signal: The transmitted signal from the transducer.
        'gaussian': A sinusoidal vibration with a Gaussian envelope.
            The frequency and bandwidth of the pulse are
            specified in the control.
        'cosine': A cosine vibration of num_periods periodes and a half period
            cosine envelope. The frequency and num_periods are specified in
            the control. The pulse is filtered around transmit frequency with
            a Gaussian filter with bandwidth specified in the control.
    :param lensfoc:
    :param pos:
    :param nofocflag:
    :return:
    """
    if lensfoc == None:
        lensfoc = numpy.zeros(2)
        if control.transducer.num_elements_azimuth == 1:
            lensfoc[0] = control.transducer.focus_azimuth
        if control.transducer.num_elements_elevation == 1:
            lensfoc[1] = control.transducer.focus_elevation

    transmit_frequency = control.signal.transmit_frequency
    p0 = control.signal.amplitude
    bandwidth = control.signal.bandwidth
    num_periods = control.signal.num_periods
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t
    resolution_x = control.signal.resolution_x
    resolution_y = control.signal.resolution_y
    resolution_t = control.signal.resolution_t
    c = control.material.material.sound_speed
    annular_transducer = control.annular_transducer

    # set length of transducer
    (idxx, idxy, _, _, _) = get_xdidx(control)
    xdnx = idxx.size
    xdny = idxy.size

    if lensfoc.size == 1:
        lensfoc = lensfoc * numpy.array([1, 1])

    # generate signal
    if isinstance(signal, str):
        if str.lower(signal) == 'gaussian':
            (t, tp, signal, nrm) = gaussian(resolution_t, transmit_frequency,
                                            num_periods, num_points_t)
        elif str.lower(signal) == 'cosine':
            t = numpy.arange(-num_points_t / 2, num_points_t / 2) * resolution_t
            Tp = num_periods / transmit_frequency
            ntp = int(numpy.round(Tp / resolution_t))
            tp = numpy.arange(-numpy.ceil(ntp / 2),
                              numpy.floor(ntp / 2) + 1) * resolution_t
            sigp = numpy.cos(
                2.0 * numpy.pi * transmit_frequency * tp) * numpy.cos(
                numpy.pi * transmit_frequency / num_periods * tp)
            signal = numpy.zeros(t.size)
            idx = numpy.where(t < tp[0])[0]
            signal[idx[-1]: idx[-1] + ntp + 1] = sigp
            (signal, _) = bandpass(signal, transmit_frequency, resolution_t,
                                   bandwidth, 4)
            nrm = numpy.max(numpy.abs(hilbert(signal)))
        else:
            print('pulse not specified - uses gaussian')
            (t, tp, signal, nrm) = gaussian(resolution_t, transmit_frequency,
                                            num_periods, num_points_t)

    signal = signal / nrm * p0

    # adjust length of signal
    if signal.size < num_points_t:
        ntzp = (num_points_t - signal.size) / 2
        signal = numpy.concatenate(
            (numpy.zeros(int((numpy.ceil(ntzp)))), signal.reshape((signal.size)),
             numpy.zeros(int((numpy.floor(ntzp))))))
    elif signal.size > 1.1 * num_points_t:
        print(
            'Increase your number of spatial grid points.\nMaximum truncation '
            'of signal vector is 10 percent.')
        exit(-1)
    else:
        nttr = int(numpy.ceil((signal.size - num_points_t) / 2.0))
        signal = signal[nttr:num_points_t + nttr]
    if control.num_dimensions == 1:
        u = signal
        return u

    # choose source
    if source == 'pointsource':
        raise NotImplementedError
    elif source == 'impulse':
        raise NotImplementedError
    elif source == 'transducer':
        # generating pulse from transducer
        control.simulation.current_position = 0

        # calculate apodization
        if isinstance(apodization, str):
            raise NotImplementedError
        elif isinstance(apodization, list):
            apodization = numpy.array(apodization)
            A = get_apodization(xdnx, xdny, 'tukey', apodization, annular_transducer)
        elif isinstance(apodization, int):
            A = get_apodization(xdnx, xdny, 'tukey', apodization, annular_transducer)
        else:
            raise NotImplementedError

        # create wave field
        xdsig = signal[..., numpy.newaxis] * A.reshape((xdnx * xdny))
        xdsig = xdsig.reshape((num_points_t, xdny, xdnx))
        xdsig, deltafoc = focus_pulse(xdsig, control, lensfoc, nofocflag)

        # create full domain
        u = numpy.zeros((num_points_t, num_points_y, num_points_x))
        u[:, slice(idxy[0], idxy[-1] + 1), slice(idxx[0], idxx[-1] + 1)] = xdsig
    else:
        print('Source type is not implemented')
        exit(-1)

    # squeeze y-direction for 2D sim
    u = numpy.squeeze(u)

    return u, deltafoc


def gaussian(resolution_t, transmit_frequency, num_periods, num_points_t):
    t = numpy.arange(-num_points_t / 2, num_points_t / 2) * resolution_t
    tp = num_periods / transmit_frequency
    sig = numpy.sin(2.0 * numpy.pi * transmit_frequency * t) * numpy.exp(
        -(2.0 * t / tp) ** 2)
    nrm = 1

    return t, tp, sig, nrm
