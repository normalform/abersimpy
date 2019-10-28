from propagation.nonlinear.get_linear import get_linear

import numpy


def burgerssolve(t,
                 u,
                 perm,
                 epsn,
                 dz):
    nt = numpy.max(t.shape)
    dt = t[1] - t[0]
    npt = (t[nt-1] - t[0]) + dt

    # introduce permutation
    t2 = t - epsn * dz * perm

    # extends by periodicity
    idt = int(numpy.floor(nt / 10))
    idxtail = numpy.arange(idt)
    idxfront = numpy.arange(nt-idt, nt)

    ttail = t2[idxtail] + npt
    tfront = t2[idxfront] - npt
    t2 = numpy.concatenate((tfront, t2, ttail))

    utail = u[idxtail]
    ufront = u[idxfront]
    u2 = numpy.concatenate((ufront, u, utail))

    # re-sample at equidistant time points
    ui = get_linear(t2, u2, t)

    return ui