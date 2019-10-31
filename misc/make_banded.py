import numpy


def make_banded(
        A,
        d=None,
        rowflag=0):
    if d is None:
        m, n = A.shape
        d = numpy.range(-m + 1, n - 1, dtype=int)

    num_points_x = numpy.min(A.shape)
    Ab = numpy.zeros((numpy.max(d.shape), num_points_x))

    for ii in range(d):
        dii = numpy.diag(A, d[ii])
        idd = numpy.range(numpy.max(dii.shape))
        stidx = numpy.max(numpy.sign(d[ii]) * (num_points_x - numpy.max(dii.shape)), 0)
        Ab[ii, idd + stidx] = dii

    if rowflag != 0:
        Ab = numpy.transpose(Ab)

    return Ab
