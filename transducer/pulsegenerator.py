import numpy
from scipy.signal import hilbert

from filter.bandpass import bandpass
from transducer.focus_pulse import focus_pulse
from transducer.get_apodization import get_apodization
from transducer.get_xdidx import get_xdidx


def pulsegenerator(prop_control,
                   src='transducer',
                   apod=0,
                   sig='gaussian',
                   lensfoc=None,
                   pos=[0, 0],
                   nofocflag=0):
    if lensfoc == None:
        lensfoc = numpy.zeros(2)
        if prop_control.num_elements_azimuth == 1:
            lensfoc[0] = prop_control.focus_azimuth
        if prop_control.num_elements_elevation == 1:
            lensfoc[1] = prop_control.focus_elevation

    transmit_frequency = prop_control.transmit_frequency
    p0 = prop_control.amplitude
    bandwidth = prop_control.bandwidth
    num_periods = prop_control.num_periods
    num_points_x = prop_control.num_points_x
    num_points_y = prop_control.num_points_y
    num_points_t = prop_control.num_points_t
    resolution_x = prop_control.resolution_x
    resolution_y = prop_control.resolution_y
    resolution_t = prop_control.resolution_t
    c = prop_control.material.sound_speed
    annular_transducer = prop_control.config.annular_transducer

    # set length of transducer
    (idxx, idxy, _, _, _) = get_xdidx(prop_control)
    xdnx = idxx.size
    xdny = idxy.size

    if lensfoc.size == 1:
        lensfoc = lensfoc * numpy.array([1, 1])

    # generate signal
    if isinstance(sig, str):
        if str.lower(sig) == 'gaussian':
            (t, tp, sig, nrm) = gaussian(resolution_t, transmit_frequency, num_periods, num_points_t)
        elif str.lower(sig) == 'cosine':
            t = numpy.arange(-num_points_t / 2, num_points_t / 2) * resolution_t
            Tp = num_periods / transmit_frequency
            ntp = int(numpy.round(Tp / resolution_t))
            tp = numpy.arange(-numpy.ceil(ntp / 2), numpy.floor(ntp / 2) + 1) * resolution_t
            sigp = numpy.cos(2.0 * numpy.pi * transmit_frequency * tp) * numpy.cos(
                numpy.pi * transmit_frequency / num_periods * tp)
            sig = numpy.zeros(t.size)
            idx = numpy.where(t < tp[0])[0]
            sig[idx[-1]: idx[-1] + ntp + 1] = sigp
            (sig, _) = bandpass(sig, transmit_frequency, resolution_t, bandwidth, 4)
            nrm = numpy.max(numpy.abs(hilbert(sig)))
        else:
            print('pulse not specified - uses gaussian')
            (t, tp, sig, nrm) = gaussian(resolution_t, transmit_frequency, num_periods, num_points_t)

    sig = sig / nrm * p0

    # adjust length of signal
    if sig.size < num_points_t:
        ntzp = (num_points_t - sig.size) / 2
        sig = numpy.concatenate(
            (numpy.zeros(int((numpy.ceil(ntzp)))), sig.reshape((sig.size)), numpy.zeros(int((numpy.floor(ntzp))))))
    elif sig.size > 1.1 * num_points_t:
        print('Increase your number of spatial grid points.\nMaximum truncation of signal vector is 10 percent.')
        exit(-1)
    else:
        nttr = int(numpy.ceil((sig.size - num_points_t) / 2.0))
        sig = sig[nttr:num_points_t + nttr]
    if prop_control.num_dimensions == 1:
        u = sig
        return u

    # choose source
    if src == 'pointsource':
        raise NotImplementedError
    elif src == 'impulse':
        raise NotImplementedError
    elif src == 'transducer':
        # generating pulse from transducer
        prop_control.current_position = 0

        # calculate apodization
        if isinstance(apod, str):
            raise NotImplementedError
        elif isinstance(apod, list):
            apod = numpy.array(apod)
            A = get_apodization(xdnx, xdny, 'tukey', apod, annular_transducer)
        elif isinstance(apod, int):
            A = get_apodization(xdnx, xdny, 'tukey', apod, annular_transducer)
        else:
            raise NotImplementedError

        # create wave field
        xdsig = sig[..., numpy.newaxis] * A.reshape((xdnx * xdny))
        xdsig = xdsig.reshape((num_points_t, xdny, xdnx))
        xdsig, deltafoc = focus_pulse(xdsig, prop_control, lensfoc, nofocflag)

        # create full domain
        u = numpy.zeros((num_points_t, num_points_y, num_points_x))
        u[:, slice(idxy[0], idxy[-1] + 1), slice(idxx[0], idxx[-1] + 1)] = xdsig
    else:
        print('Source type is not implemented')
        exit(-1)

    # squeeze y-direction for 2D sim
    u = numpy.squeeze(u)

    return u, prop_control, deltafoc


def gaussian(resolution_t, transmit_frequency, num_periods, num_points_t):
    t = numpy.arange(-num_points_t / 2, num_points_t / 2) * resolution_t
    tp = num_periods / transmit_frequency
    sig = numpy.sin(2.0 * numpy.pi * transmit_frequency * t) * numpy.exp(-(2.0 * t / tp) ** 2)
    nrm = 1

    return t, tp, sig, nrm
