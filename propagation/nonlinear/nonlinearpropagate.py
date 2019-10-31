import numpy

import propagation
from controls.consts import ScaleForSpatialVariablesZ, ScaleForTemporalVariable
from controls.main_control import NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from misc.make_banded import make_banded
from propagation.get_diffmatrix import get_diffmatrix
from propagation.get_wavenumbers import get_wavenumbers
from propagation.nonlinear.nonlinattenuationsplit import nonlinattenuationsplit


def nonlinearpropagate(u_z,
                       dir,
                       main_control=None,
                       kz=None,
                       eps_n=None,
                       eps_a=None,
                       eps_b=None):
    global KZ
    if main_control is None:
        print('Wave field, direction and main_control must be specified')
        exit(-1)
    if kz is not None:
        KZ = kz
        del kz

    # initialization
    if KZ.size == 0:
        KZ = get_wavenumbers(main_control)

    # preparation of variables
    mat = main_control.material
    c = mat.sound_speed
    c = c / ScaleForSpatialVariablesZ * ScaleForTemporalVariable  # scaling wave-speed
    resolution_t = main_control.resolution_t / ScaleForTemporalVariable  # scale sampling to microsecs
    resolution_x = main_control.resolution_x / ScaleForSpatialVariablesZ  # resolution_x scaled to centimeter
    resolution_y = main_control.resolution_y / ScaleForSpatialVariablesZ  # resolution_y scaled to centimeter
    resolution_z = main_control.resolution_z / ScaleForSpatialVariablesZ  # resolution_z scaled to centimeter
    num_dimensions = main_control.num_dimensions
    num_points_x = main_control.num_points_x
    num_points_y = main_control.num_points_y
    num_points_t = main_control.num_points_t

    annular_transducer = main_control.config.annular_transducer
    shock_step = main_control.shock_step
    step_size = main_control.step_size
    perfect_matching_layer_width = main_control.perfect_matching_layer_width

    nsubsteps = int(numpy.ceil((step_size / ScaleForSpatialVariablesZ) / resolution_z))
    resolution_z = (step_size / ScaleForSpatialVariablesZ) / nsubsteps
    d = (c / 2) * (resolution_t / 2) * resolution_z
    tspan = numpy.transpose(numpy.linspace(resolution_t, num_points_t * resolution_t + resolution_t, num_points_t))

    # assign flags
    diffraction_type = main_control.config.diffraction_type
    non_linearity = main_control.config.non_linearity
    attenuation = main_control.config.attenuation

    # prepare PML and FD matrices
    if perfect_matching_layer_width > 0:
        A = d * get_diffmatrix(2 * perfect_matching_layer_width, resolution_x, 4)
    if diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
            diffraction_type == FiniteDifferenceTimeDifferenceFull:
        if numpy.abs(KZ[4] - d) > 1e-12:
            # find difference matrix A
            Ax = d * get_diffmatrix(num_points_x, resolution_x, 4, annular_transducer)
            Bx = numpy.eye(num_points_x) + Ax
            Dx = numpy.eye(num_points_x) - Ax
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
        main_control.step_size = resolution_z * ScaleForSpatialVariablesZ

    # Nonlinear propagation
    for ni in range(nsubsteps):
        # diffraction
        if diffraction_type == ExactDiffraction or \
                diffraction_type == AngularSpectrumDiffraction or \
                diffraction_type == PseudoDifferential:
            u_z, _ = propagation.propagate.propagate(u_z, 2 * dir, main_control, KZ)
        elif diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
                diffraction_type == FiniteDifferenceTimeDifferenceFull:
            raise NotImplementedError

        # perfectly matching layers, absorbing boundaries
        if perfect_matching_layer_width > 0:
            if propagation.num_dimensions == 2:
                u_z = u_z.reshape((num_points_x, 1, num_points_t))
                raise NotImplementedError

        # Nonlinear and attenuation
        if non_linearity or attenuation:
            u_z = nonlinattenuationsplit(tspan, u_z, resolution_z, shock_step, mat, non_linearity, attenuation)

    # set step_size back to normal
    if diffraction_type == ExactDiffraction or diffraction_type == AngularSpectrumDiffraction:
        main_control.step_size = step_size

    return u_z
