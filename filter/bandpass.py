from filter.get_freqs import get_freqs

import numpy
import scipy.sparse


def bandpass(u,
             fc,
             dt,
             bwidth = None,
             par = 4,
             db = -6):
    filt = 0
    if bwidth is not None and bwidth == numpy.inf:
        return u, filt

    # find size of and reshape u
    nt = u.shape[0]
    ny = 1
    nx = 1
    ndim = u.ndim
    if ndim == 2:
        nx = u.shape[1]
        u = u.reshape((nt, nx*ny))

    # performs filtering in the frequency domain
    U = numpy.fft.fftn(u, axes=(0,))
    f = get_freqs(nt, dt)

    # find the center frequency if fc is an interval
    if isinstance(fc, float):
        fc = numpy.array([fc])
    if fc.ndim == 2:
        raise NotImplementedError('Need to verify')
        idxf = numpy.where(f >= fc[0] and f <= fc[1])[0]
        if idxf.size == 0:
            fc0 = numpy.mean(fc)
        else:
            tmpU = numpy.mean(numpy.abs(U[idxf, :]), axis=1)
            idxfc = numpy.argmax(tmpU, axis=0)
            fc0 = numpy.mean(f[idxf[idxfc]])
    else:
        fc0 = fc[0]

    if bwidth is None:
        bwidth = 0.5 * fc0
    elif bwidth < 2.0:
        bwidth = bwidth * fc0

    a = numpy.log(10 ** (db / 20.0))
    filt = numpy.exp(a * (numpy.abs(numpy.abs(f) - fc0) / (bwidth / 2.0)) ** par)
    dfilt = scipy.sparse.spdiags(filt, 0, nt, nt)
    U = dfilt * U
    u = numpy.fft.ifftn(U, axes=(0,)).real

    if ndim == 2:
        u = u.reshape((nt, ny, nx))

    return u, filt