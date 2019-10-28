from propagation.get_diffstencil import get_diffstencil

import numpy
import scipy.sparse


def get_diffmatrix(N = 8,
                   h = 1,
                   order = 4,
                   annflag = 0,
                   leftbc = 'symmetric'):
    col = numpy.ones((N, 1))
    d = get_diffstencil(order, 2)
    A = scipy.sparse.spdiags(col * d, numpy.arange(-order/2, order/2, dtype=int), N, N)
    raise NotImplementedError