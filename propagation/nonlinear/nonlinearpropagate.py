from propagation.get_wavenumbers import get_wavenumbers
from propagation.get_diffmatrix import get_diffmatrix
from propagation.nonlinear.nonlinattenuationsplit import nonlinattenuationsplit
import propagation
from misc.make_banded import make_banded
from consts import ZSCALE, TSCALE

import numpy


def nonlinearpropagate(u_z,
                       dir,
                       propcontrol = None,
                       kz = None,
                       eps_n = None,
                       eps_a = None,
                       eps_b = None):
    global KZ
    if propcontrol is None:
        print('Wave field, direction and propcontrol must be specified')
        exit(-1)
    if kz is not None:
        KZ = kz
        del kz

    # initialization
    if KZ.size == 0:
        KZ = get_wavenumbers(propcontrol)

    # preparation of variables
    mat = propcontrol.material
    c = mat.c0
    c = c / ZSCALE * TSCALE         # scaling wave-speed
    dt = propcontrol.dt / TSCALE    # scale sampling to microsecs
    dx = propcontrol.dx / ZSCALE    # dx scaled to centimeter
    dy = propcontrol.dy / ZSCALE    # dy scaled to centimeter
    dz = propcontrol.dz / ZSCALE    # dz scaled to centimeter
    ndim = propcontrol.ndims
    nx = propcontrol.nx
    ny = propcontrol.ny
    nt = propcontrol.nt

    annflag = propcontrol.annflag
    shockstep = propcontrol.shockstep
    stepsize = propcontrol.stepsize
    PMLwidth = propcontrol.PMLwidth

    nsubsteps = int(numpy.ceil((stepsize / ZSCALE) / dz))
    dz = (stepsize / ZSCALE) / nsubsteps
    d = (c / 2) * (dt / 2) * dz
    tspan = numpy.transpose(numpy.linspace(dt, nt * dt + dt, nt))

    # assign flags
    diffrflag = propcontrol.diffrflag
    nonlinflag = propcontrol.nonlinflag
    lossflag = propcontrol.lossflag

    # prepare PML and FD matrices
    if PMLwidth > 0:
        A = d * get_diffmatrix(2 * PMLwidth, dx, 4)
    if diffrflag > 3:
        if numpy.abs(KZ[4] - d) > 1e-12:
            # find difference matrix A
            Ax = d * get_diffmatrix(nx, dx, 4, annflag)
            Bx = numpy.eye(nx) + Ax
            Dx = numpy.eye(nx) - Ax
            Dxi = numpy.inv(Dx)

            if diffrflag == 4:
                # Make matrices banded
                Bxb = make_banded(Bx, numpy.arange(-2, 2, dtype=int))
                Dxib = make_banded(Dxi, numpy.arange(-10, 10, dtype=int))
            elif diffrflag == 5:
                raise NotImplementedError
    elif diffrflag <= 3:
        propcontrol.stepsize = dz * ZSCALE

    # Nonlinear propagation
    for ni in range(nsubsteps):
        # diffraction
        if diffrflag <= 3 and diffrflag > 0:
            u_z, _ = propagation.propagate.propagate(u_z, 2 * dir, propcontrol, KZ)
        elif diffrflag == 4 or diffrflag == 5:
            raise NotImplementedError

        # perfectly matching layers, absorbing boundaries
        if PMLwidth > 0:
            if propagation.ndims == 2:
                u_z = u_z.reshape((nx, 1, nt))
                raise NotImplementedError

        # Nonlinear and attenuation
        if nonlinflag != 0 or lossflag != 0:
            u_z = nonlinattenuationsplit(tspan, u_z, dz, shockstep, mat, nonlinflag, lossflag)

    # set stepsize back to normal
    if diffrflag == 1 or diffrflag == 2:
        propcontrol.stepsize = stepsize

    return u_z