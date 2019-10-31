import numpy
import scipy.sparse

from filter.get_freqs import get_freqs


def bandpass(u,
             transmit_frequency,
             resolution_t,
             bwidth=None,
             par=4,
             db=-6):
    filt = 0
    if bwidth is not None and bwidth == numpy.inf:
        return u, filt

    # find size of and reshape u
    num_points_t = u.shape[0]
    num_points_y = 1
    num_points_x = 1
    num_dimensions = u.ndim
    if num_dimensions == 2:
        num_points_x = u.shape[1]
        u = u.reshape((num_points_t, num_points_x * num_points_y))

    # performs filtering in the frequency domain
    U = numpy.fft.fftn(u, axes=(0,))
    f = get_freqs(num_points_t, resolution_t)

    # find the center frequency if transmit_frequency is an interval
    if isinstance(transmit_frequency, float):
        transmit_frequency = numpy.array([transmit_frequency])
    if transmit_frequency.ndim == 2:
        raise NotImplementedError('Need to verify')
        idxf = numpy.where(f >= transmit_frequency[0] and f <= transmit_frequency[1])[0]
        if idxf.size == 0:
            fc0 = numpy.mean(transmit_frequency)
        else:
            tmpU = numpy.mean(numpy.abs(U[idxf, :]), axis=1)
            idxfc = numpy.argmax(tmpU, axis=0)
            fc0 = numpy.mean(f[idxf[idxfc]])
    else:
        fc0 = transmit_frequency[0]

    if bwidth is None:
        bwidth = 0.5 * fc0
    elif bwidth < 2.0:
        bwidth = bwidth * fc0

    a = numpy.log(10 ** (db / 20.0))
    filt = numpy.exp(a * (numpy.abs(numpy.abs(f) - fc0) / (bwidth / 2.0)) ** par)
    dfilt = scipy.sparse.spdiags(filt, 0, num_points_t, num_points_t)
    U = dfilt * U
    u = numpy.fft.ifftn(U, axes=(0,)).real

    if num_dimensions == 2:
        u = u.reshape((num_points_t, num_points_y, num_points_x))

    return u, filt
