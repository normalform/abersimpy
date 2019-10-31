import numpy

from propagation.nonlinear.get_linear import get_linear


def burgerssolve(t,
                 u,
                 perm,
                 epsn,
                 resolution_z):
    num_points_t = numpy.max(t.shape)
    resolution_t = t[1] - t[0]
    npt = (t[num_points_t - 1] - t[0]) + resolution_t

    # introduce permutation
    t2 = t - epsn * resolution_z * perm

    # extends by periodicity
    idt = int(numpy.floor(num_points_t / 10))
    idxtail = numpy.arange(idt)
    idxfront = numpy.arange(num_points_t - idt, num_points_t)

    ttail = t2[idxtail] + npt
    tfront = t2[idxfront] - npt
    t2 = numpy.concatenate((tfront, t2, ttail))

    utail = u[idxtail]
    ufront = u[idxfront]
    u2 = numpy.concatenate((ufront, u, utail))

    # re-sample at equidistant time points
    ui = get_linear(t2, u2, t)

    return ui
