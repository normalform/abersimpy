import numpy


def get_rms(u_z,
            scaleflag=1.0):
    if scaleflag != 0:
        N = u_z.shape[0]
        rmspro = numpy.sqrt(1 / N * numpy.sum(u_z ** 2, axis=0))
    else:
        rmspro = numpy.sqrt(numpy.sum(u_z ** 2, axis=0))

    return rmspro
