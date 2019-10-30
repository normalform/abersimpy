import numpy

import propagation
from consts import ScaleForSpatialVariablesZ, ScaleForTemporalVariable
from misc.make_banded import make_banded
from prop_control import NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from propagation.get_diffmatrix import get_diffmatrix
from propagation.get_wavenumbers import get_wavenumbers
from propagation.nonlinear.nonlinattenuationsplit import nonlinattenuationsplit


def nonlinearpropagate(u_z,
                       dir,
                       prop_control=None,
                       kz = None,
                       eps_n = None,
                       eps_a = None,
                       eps_b = None):
    global KZ
    if prop_control is None:
        print('Wave field, direction and prop_control must be specified')
        exit(-1)
    if kz is not None:
        KZ = kz
        del kz

    # initialization
    if KZ.size == 0:
        KZ = get_wavenumbers(prop_control)

    # preparation of variables
    mat = prop_control.material
    c = mat.sound_speed
    c = c / ScaleForSpatialVariablesZ * ScaleForTemporalVariable  # scaling wave-speed
    dt = prop_control.dt / ScaleForTemporalVariable  # scale sampling to microsecs
    dx = prop_control.dx / ScaleForSpatialVariablesZ  # dx scaled to centimeter
    dy = prop_control.dy / ScaleForSpatialVariablesZ  # dy scaled to centimeter
    dz = prop_control.dz / ScaleForSpatialVariablesZ  # dz scaled to centimeter
    num_dimensions = prop_control.num_dimensions
    nx = prop_control.nx
    ny = prop_control.ny
    nt = prop_control.nt

    annular_transducer = prop_control.config.annular_transducer
    shockstep = prop_control.shockstep
    stepsize = prop_control.stepsize
    PMLwidth = prop_control.PMLwidth

    nsubsteps = int(numpy.ceil((stepsize / ScaleForSpatialVariablesZ) / dz))
    dz = (stepsize / ScaleForSpatialVariablesZ) / nsubsteps
    d = (c / 2) * (dt / 2) * dz
    tspan = numpy.transpose(numpy.linspace(dt, nt * dt + dt, nt))

    # assign flags
    diffraction_type = prop_control.config.diffraction_type
    non_linearity = prop_control.config.non_linearity
    attenuation = prop_control.config.attenuation

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
        prop_control.stepsize = dz * ScaleForSpatialVariablesZ

    # Nonlinear propagation
    for ni in range(nsubsteps):
        # diffraction
        if diffraction_type == ExactDiffraction or \
            diffraction_type == AngularSpectrumDiffraction or \
            diffraction_type == PseudoDifferential:
            u_z, _ = propagation.propagate.propagate(u_z, 2 * dir, prop_control, KZ)
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
        prop_control.stepsize = stepsize

    return u_z