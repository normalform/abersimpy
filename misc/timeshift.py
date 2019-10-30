from filter.get_freqs import get_freqs

import numpy


def timeshift(u,
              delta,
              method = 'fft'):
    num_dimensions = u.ndim
    (nt, ny, nx) = u.shape

    if num_dimensions == 2:
        nch = ny
    else:
        nch = nx * ny
        u = u.reshape((nt, nch))

    delta = delta.reshape((1, delta.size))
    if numpy.max(delta.shape) != nch and numpy.max(delta.shape) > 1:
        raise NotImplementedError

    if method == 'fft':
        u = numpy.fft.fftn(u, axes=(0,))
        k = get_freqs(nt, 1)[..., numpy.newaxis]
        if numpy.max(delta.shape) == 1:
            delta = numpy.ones(nch) * delta
        sh = numpy.exp(-1j * 2 * numpy.pi * k * delta)
        u = u * sh
        u = numpy.fft.ifftn(u, axes=(0,)).real
    else:
        raise NotImplementedError

    u = u.reshape(nt, ny, nx)
    return u
