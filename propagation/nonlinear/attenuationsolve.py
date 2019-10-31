import numpy
from scipy.signal import hilbert

from filter.get_freqs import get_freqs


def attenuationsolve(t,
                     ui,
                     resolution_z,
                     epsa,
                     epsb):
    resolution_t = t[1] - t[0]
    ui = numpy.array(ui)
    num_points_t = numpy.max(ui.shape)
    loss = 2 * numpy.pi * get_freqs(num_points_t, resolution_t)

    # prepare attenuation coefficients
    loss = epsa * numpy.conj(hilbert(numpy.abs(loss) ** epsb)) * resolution_z
    loss = numpy.exp(-loss)

    U = numpy.fft.fftn(ui, axes=(0,))
    U = loss * U
    u = numpy.fft.ifftn(U, axes=(0,)).real

    return u
