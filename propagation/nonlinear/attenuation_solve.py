"""
attenuation_solve.py
"""
import numpy
from scipy.signal import hilbert

from filter.get_frequencies import get_frequencies


def attenuation_solve(t,
                      ui,
                      resolution_z,
                      eps_a,
                      eps_b):
    resolution_t = t[1] - t[0]
    ui = numpy.array(ui)
    num_points_t = numpy.max(ui.shape)
    loss = 2 * numpy.pi * get_frequencies(num_points_t, resolution_t)

    # prepare attenuation coefficients
    loss = eps_a * numpy.conj(hilbert(numpy.abs(loss) ** eps_b)) * resolution_z
    loss = numpy.exp(-loss)

    U = numpy.fft.fftn(ui, axes=(0,))
    U = loss * U
    u = numpy.fft.ifftn(U, axes=(0,)).real

    return u
