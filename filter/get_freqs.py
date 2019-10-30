import numpy


def get_freqs(N, dt):
    df = 1.0 / (dt * N)
    f = numpy.fft.ifftshift(numpy.arange(0, N) - numpy.floor(N / 2)) * df

    return f
