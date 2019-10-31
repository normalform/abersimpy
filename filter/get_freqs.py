import numpy


def get_freqs(N, resolution_t):
    df = 1.0 / (resolution_t * N)
    f = numpy.fft.ifftshift(numpy.arange(0, N) - numpy.floor(N / 2)) * df

    return f
