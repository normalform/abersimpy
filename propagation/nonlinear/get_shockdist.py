import numpy


def get_shockdist(t,
                  p,
                  epsn):
    dt = numpy.diff(t)
    if numpy.min(dt) <= 0.0:
        return 0.0

    # calculate shock distance
    dpdt_max = numpy.max(numpy.diff(p) / dt)
    if dpdt_max > 0.0 and epsn != 0.0:
        shockdist = 1 / (epsn * dpdt_max)
    else:
        shockdist = numpy.inf

    return shockdist
