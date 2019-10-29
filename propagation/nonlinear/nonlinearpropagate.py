from propagation.get_wavenumbers import get_wavenumbers
from propagation.get_diffmatrix import get_diffmatrix
from propagation.nonlinear.nonlinattenuationsplit import nonlinattenuationsplit
import propagation
from misc.make_banded import make_banded
from consts import ZSCALE, TSCALE
from propcontrol import NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull

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
    num_dimensions = propcontrol.num_dimensions
    nx = propcontrol.nx
    ny = propcontrol.ny
    nt = propcontrol.nt

    annular_transducer = propcontrol.config.annular_transducer
    shockstep = propcontrol.shockstep
    stepsize = propcontrol.stepsize
    PMLwidth = propcontrol.PMLwidth

    nsubsteps = int(numpy.ceil((stepsize / ZSCALE) / dz))
    dz = (stepsize / ZSCALE) / nsubsteps
    d = (c / 2) * (dt / 2) * dz
    tspan = numpy.transpose(numpy.linspace(dt, nt * dt + dt, nt))

    # assign flags
    diffraction_type = propcontrol.config.diffraction_type
    non_linearity = propcontrol.config.non_linearity
    attenuation = propcontrol.config.attenuation

    # prepare PML and FD matrices
    if PMLwidth > 0:
        A = d * get_diffmatrix(2 * PMLwidth, dx, 4)
    if diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
            diffraction_type == FiniteDifferenceTimeDifferenceFull:
        if numpy.abs(KZ[4] - d) > 1e-12:
            # find difference matrix A
            Ax = d * get_diffmatrix(nx, dx, 4, annular_transducer)
            Bx = numpy.eye(nx) + Ax
            Dx = numpy.eye(nx) - Ax
            Dxi = numpy.inv(Dx)

            if diffraction_type == FiniteDifferenceTimeDifferenceReduced:
                # Make matrices banded
                Bxb = make_banded(Bx, numpy.arange(-2, 2, dtype=int))
                Dxib = make_banded(Dxi, numpy.arange(-10, 10, dtype=int))
            elif diffraction_type == FiniteDifferenceTimeDifferenceFull:
                raise NotImplementedError
    elif diffraction_type == NoDiffraction or \
            diffraction_type == ExactDiffraction or \
            diffraction_type == AngularSpectrumDiffraction or \
            diffraction_type == PseudoDifferential:
        propcontrol.stepsize = dz * ZSCALE

    # Nonlinear propagation
    for ni in range(nsubsteps):
        # diffraction
        if diffraction_type == ExactDiffraction or \
            diffraction_type == AngularSpectrumDiffraction or \
            diffraction_type == PseudoDifferential:
            u_z, _ = propagation.propagate.propagate(u_z, 2 * dir, propcontrol, KZ)
        elif diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
                diffraction_type == FiniteDifferenceTimeDifferenceFull:
            raise NotImplementedError

        # perfectly matching layers, absorbing boundaries
        if PMLwidth > 0:
            if propagation.num_dimensions == 2:
                u_z = u_z.reshape((nx, 1, nt))
                raise NotImplementedError

        # Nonlinear and attenuation
        if non_linearity or attenuation:
            u_z = nonlinattenuationsplit(tspan, u_z, dz, shockstep, mat, non_linearity, attenuation)

    # set stepsize back to normal
    if diffraction_type == ExactDiffraction or diffraction_type == AngularSpectrumDiffraction:
        propcontrol.stepsize = stepsize

    return u_z