import numpy
from scipy.signal import hilbert


def get_max(u_z,
            envflag = 0):

    if envflag != 0:
        maxpro = numpy.max(numpy.abs(hilbert(u_z, axis=0)), axis=0)
    else:
        maxpro = numpy.max(u_z, axis=0)
    return maxpro
