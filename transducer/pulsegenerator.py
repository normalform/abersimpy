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
        if prop_control.Nex == 1:
            lensfoc[0] = prop_control.Fx
        if prop_control.Ney == 1:
            lensfoc[1] = prop_control.Fy

    fc = prop_control.fc
    p0 = prop_control.amplitude
    bandwidth = prop_control.bandwidth
    np = prop_control.np
    nx = prop_control.nx
    ny = prop_control.ny
    nt = prop_control.nt
    dx = prop_control.dx
    dy = prop_control.dy
    dt = prop_control.dt
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
            (t, tp, sig, nrm) = gaussian(dt, fc, np, nt)
        elif str.lower(sig) == 'cosine':
            t = numpy.arange(-nt / 2, nt / 2) * dt
            Tp = np / fc
            ntp = int(numpy.round(Tp / dt))
            tp = numpy.arange(-numpy.ceil(ntp / 2), numpy.floor(ntp / 2) + 1) * dt
            sigp = numpy.cos(2.0 * numpy.pi * fc * tp) * numpy.cos(numpy.pi * fc / np * tp)
            sig = numpy.zeros(t.size)
            idx = numpy.where(t < tp[0])[0]
            sig[idx[-1]: idx[-1] + ntp + 1] = sigp
            (sig, _) = bandpass(sig, fc, dt, bandwidth, 4)
            nrm = numpy.max(numpy.abs(hilbert(sig)))
        else:
            print('pulse not specified - uses gaussian')
            (t, tp, sig, nrm) = gaussian(dt, fc, np, nt)

    sig = sig / nrm * p0

    # adjust length of signal
    if sig.size < nt:
        ntzp = (nt - sig.size) / 2
        sig = numpy.concatenate(
            (numpy.zeros(int((numpy.ceil(ntzp)))), sig.reshape((sig.size)), numpy.zeros(int((numpy.floor(ntzp))))))
    elif sig.size > 1.1 * nt:
        print('Increase your number of spatial grid points.\nMaximum truncation of signal vector is 10 percent.')
        exit(-1)
    else:
        nttr = int(numpy.ceil((sig.size - nt) / 2.0))
        sig = sig[nttr:nt + nttr]
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
        prop_control.currentpos = 0

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
        xdsig = xdsig.reshape((nt, xdny, xdnx))
        xdsig, deltafoc = focus_pulse(xdsig, prop_control, lensfoc, nofocflag)

        # create full domain
        u = numpy.zeros((nt, ny, nx))
        u[:, slice(idxy[0], idxy[-1] + 1), slice(idxx[0], idxx[-1] + 1)] = xdsig
    else:
        print('Source type is not implemented')
        exit(-1)

    # squeeze y-direction for 2D sim
    u = numpy.squeeze(u)

    return u, prop_control, deltafoc


def gaussian(dt, fc, np, nt):
    t = numpy.arange(-nt / 2, nt / 2) * dt
    tp = np / fc
    sig = numpy.sin(2.0 * numpy.pi * fc * t) * numpy.exp(-(2.0 * t / tp) ** 2)
    nrm = 1

    return t, tp, sig, nrm
