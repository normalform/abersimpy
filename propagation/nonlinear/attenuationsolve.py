from filter.get_freqs import get_freqs
from scipy.signal import hilbert

import numpy
import matplotlib.pyplot as plt


def attenuationsolve(t,
                     ui,
                     dz,
                     epsa,
                     epsb):
    dt = t[1] - t[0]
    ui = numpy.array(ui)
    nt= numpy.max(ui.shape)
    loss = 2 * numpy.pi * get_freqs(nt, dt)

    # prepare attenuation coefficients
    loss = epsa * numpy.conj(hilbert(numpy.abs(loss) ** epsb)) * dz
    loss = numpy.exp(-loss)

    U = numpy.fft.fftn(ui, axes=(0,))
    U = loss * U
    u = numpy.fft.ifftn(U, axes=(0,)).real

    return u
