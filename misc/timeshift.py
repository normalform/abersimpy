import numpy

from filter.get_freqs import get_freqs


def timeshift(u,
              delta,
              method='fft'):
    num_dimensions = u.ndim
    (num_points_t, num_points_y, num_points_x) = u.shape

    if num_dimensions == 2:
        nch = num_points_y
    else:
        nch = num_points_x * num_points_y
        u = u.reshape((num_points_t, nch))

    delta = delta.reshape((1, delta.size))
    if numpy.max(delta.shape) != nch and numpy.max(delta.shape) > 1:
        raise NotImplementedError

    if method == 'fft':
        u = numpy.fft.fftn(u, axes=(0,))
        k = get_freqs(num_points_t, 1)[..., numpy.newaxis]
        if numpy.max(delta.shape) == 1:
            delta = numpy.ones(nch) * delta
        sh = numpy.exp(-1j * 2 * numpy.pi * k * delta)
        u = u * sh
        u = numpy.fft.ifftn(u, axes=(0,)).real
    else:
        raise NotImplementedError

    u = u.reshape(num_points_t, num_points_y, num_points_x)
    return u
