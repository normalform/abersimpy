# -*- coding: utf-8 -*-
"""
    nonlinear_propagate.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import numpy
import scipy.sparse

from simulation.controls.consts import SCALE_FOR_SPATIAL_VARIABLES_Z, SCALE_FOR_TEMPORAL_VARIABLE
from simulation.controls.main_control import MainControl
from simulation.get_wave_numbers import get_wave_numbers
from simulation.propagation import propagate
from simulation.propagation.nonlinear.attenuation_solve import attenuation_solve
from simulation.propagation.nonlinear.burgers_solve import burgers_solve
from system.diffraction.diffraction import NoDiffraction, ExactDiffraction, \
    AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from system.material.muscle import Muscle


def nonlinear_propagate(control: MainControl,
                        wave: numpy.ndarray,
                        direction: int,
                        equidistant_steps,
                        wave_numbers=None,
                        eps_n=None,
                        eps_a=None,
                        eps_b=None):
    """
    Handles nonlinear propagation of 3D wave field in z-direction
    using an operator splitting method.
    :param control: The Controls.
    :param wave: Wave at initial position.
    :param direction: Direction of propagation.
        1 - positive z-direction
        -1 - negative z-direction
    :param equidistant_steps:
    :param wave_numbers: 3D wave numbers. Default is given by the control with retarded time.
    :param eps_n: The parameter governing non-linearity.
        Default is given by the material specified in the control.
    :param eps_a: Used to specify frequency dependant loss.
    :param eps_b: Used to specify frequency dependant loss.
    :return: The resulting field after propagation.
    """
    # initialization
    if wave_numbers is None:
        _wave_numbers = get_wave_numbers(control, equidistant_steps)
    else:
        _wave_numbers = wave_numbers

    if _wave_numbers.size == 0:
        _wave_numbers = get_wave_numbers(control, equidistant_steps)

    # preparation of variables
    material = control.material.material

    # scaling sound speed
    sound_speed = material.sound_speed / SCALE_FOR_SPATIAL_VARIABLES_Z * SCALE_FOR_TEMPORAL_VARIABLE
    # scale sampling to micro seconds
    resolution_t = control.signal.resolution_t / SCALE_FOR_TEMPORAL_VARIABLE
    # resolution_x scaled to centimeter
    resolution_x = control.signal.resolution_x / SCALE_FOR_SPATIAL_VARIABLES_Z
    # resolution_y scaled to centimeter
    resolution_y = control.signal.resolution_y / SCALE_FOR_SPATIAL_VARIABLES_Z
    # resolution_z scaled to centimeter
    resolution_z = control.signal.resolution_z / SCALE_FOR_SPATIAL_VARIABLES_Z

    num_dimensions = control.num_dimensions
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t

    annular_transducer = control.annular_transducer
    shock_step = control.simulation.shock_step
    step_size = control.simulation.step_size
    perfect_matching_layer_width = control.domain.perfect_matching_layer_width

    num_sub_steps = int(numpy.ceil((step_size / SCALE_FOR_SPATIAL_VARIABLES_Z) / resolution_z))
    resolution_z = (step_size / SCALE_FOR_SPATIAL_VARIABLES_Z) / num_sub_steps
    d = (sound_speed / 2) * (resolution_t / 2) * resolution_z
    t_span = numpy.linspace(resolution_t,
                            num_points_t *
                            resolution_t + resolution_t,
                            num_points_t).T

    # assign flags
    diffraction_type = control.diffraction_type
    non_linearity = control.non_linearity
    attenuation = control.attenuation

    # prepare PML and FD matrices
    if perfect_matching_layer_width > 0:
        a = d * _get_difference_matrix(2 * perfect_matching_layer_width,
                                       resolution_x,
                                       4)

    if diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
            diffraction_type == FiniteDifferenceTimeDifferenceFull:
        if numpy.abs(_wave_numbers[4] - d) > 1e-12:
            # find difference matrix A
            ax = d * _get_difference_matrix(num_points_x,
                                            resolution_x,
                                            4,
                                            annular_transducer)
            bx = numpy.eye(num_points_x) + ax
            dx = numpy.eye(num_points_x) - ax
            dx_inv = numpy.inv(dx)

            if diffraction_type == FiniteDifferenceTimeDifferenceReduced:
                # Make matrices banded
                bx_banded = _make_banded(bx, numpy.arange(-2, 2, dtype=int))
                dx_inv_banded = _make_banded(dx_inv, numpy.arange(-10, 10, dtype=int))
            elif diffraction_type == FiniteDifferenceTimeDifferenceFull:
                raise NotImplementedError
    elif diffraction_type in (NoDiffraction,
                              ExactDiffraction,
                              AngularSpectrumDiffraction,
                              PseudoDifferential):
        control.simulation.step_size = resolution_z * SCALE_FOR_SPATIAL_VARIABLES_Z

    _wave = wave

    # Nonlinear propagation
    for index in range(num_sub_steps):
        # diffraction
        if diffraction_type in (ExactDiffraction,
                                AngularSpectrumDiffraction,
                                PseudoDifferential):
            _wave = propagate.propagate(control,
                                        _wave,
                                        2 * direction,
                                        equidistant_steps,
                                        _wave_numbers)
        elif diffraction_type in (FiniteDifferenceTimeDifferenceReduced,
                                  FiniteDifferenceTimeDifferenceFull):
            raise NotImplementedError

        # perfectly matching layers, absorbing boundaries
        if perfect_matching_layer_width > 0:
            if control.num_dimensions == 2:
                _wave = _wave.reshape((num_points_x, 1, num_points_t))
                raise NotImplementedError

        # Nonlinear and attenuation
        if non_linearity or attenuation:
            _wave = _nonlinear_attenuation_split(t_span,
                                                 _wave,
                                                 resolution_z,
                                                 shock_step,
                                                 material,
                                                 non_linearity,
                                                 attenuation)

    # set step_size back to normal
    if diffraction_type in (ExactDiffraction, AngularSpectrumDiffraction):
        control.simulation.step_size = step_size

    return _wave


def _get_difference_stencil(num_order: int = 4,
                            differential_order: int = 2):
    """
    Returns weights for a difference stencil.
    :param num_order: Numerical order of the difference scheme.
        Default is forth order central differencing.
    :param differential_order: Differential order. Default is the stencil for the
        second order differential. First order is possible.
    :return: Difference stencil as a row vector.
    """
    if differential_order == 1:
        if num_order == 2:
            difference_stencil = numpy.array([-1.0, 0.0, 1.0]) / 2.0
        elif num_order == 4:
            difference_stencil = numpy.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
        else:
            print('Unrecognized numerical order. Using fourth order differencing')
            difference_stencil = _get_difference_stencil(differential_order, 4)
    elif differential_order == 2:
        if num_order == 2:
            difference_stencil = numpy.array([-1.0, 2.0, -1.0])
        elif num_order == 4:
            difference_stencil = numpy.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
        else:
            print('Unrecognized numerical order. Using fourth order differencing')
            difference_stencil = _get_difference_stencil(differential_order, 4)
    else:
        print('Unrecognized differential order, Using second order differential')
        difference_stencil = _get_difference_stencil(2, num_order)

    return difference_stencil


def _get_difference_matrix(num_points: int = 8,
                           resolution: int = 1,
                           order: int = 4,
                           annular_transducer: bool = False,
                           left_boundary='symmetric'):
    """
    The function returns a differencing matrix based on a finite difference
    scheme of desired order. The laplacian may either be d^2/dx^2 or d^2/dr^2+1/r d/dr.
    :param num_points: Number of points. Default 8 for example purposes.
    :param resolution: Resolution in space. Default is one for example purposes.
    :param order: Flag for rotational symmetry. Default is False.
    :param annular_transducer:
    :param left_boundary: The left boundary may be either 'symmetric' or 'reflective'.
        The outer (right) boundary is assumed to always be reflective.
    :return: Difference matrix.
    """
    col = numpy.ones((num_points, 1))
    d = _get_difference_stencil(order, 2)
    a = scipy.sparse.spdiags(col * d,
                             numpy.arange(-order / 2, order / 2, dtype=int),
                             num_points,
                             num_points)
    raise NotImplementedError


def _nonlinear_attenuation_split(scaled_time_span,
                                 wave_field,
                                 resolution_z,
                                 shock_step,
                                 material=Muscle(37),
                                 non_linearity=True,
                                 attenuation=True):
    """
    Implements a loop over the spatial direction with calls to BurgersSplit.
    :param scaled_time_span: Scaled time span
    :param wave_field: Wave field.
    :param resolution_z: resolution_z.
    :param shock_step: Shock step.
    :param material: Material used. If not specified, MUSCLE is assumed.
    :param non_linearity: non linearity.
    :param attenuation: attenuation.
    :return: Perturbed and attenuated wave field.
    """
    # Initiation of sizes
    num_dimensions = wave_field.ndim
    if num_dimensions == 3:
        num_points_t, num_points_y, num_points_x = wave_field.shape
        _wave_field = wave_field.reshape((num_points_t, num_points_x * num_points_y))
    else:
        num_points_t, num_points_x = wave_field.shape
        num_points_y = 1
        _wave_field = wave_field

    # Check material
    is_regular = material.is_regular
    eps_n = material.eps_n
    eps_a = material.eps_a
    eps_b = material.eps_b

    for index in range(num_points_x * num_points_y):
        num_step = 0
        temp = _wave_field[:, index]
        dz_tmp = resolution_z
        while dz_tmp > 0:
            z_step = numpy.minimum(dz_tmp,
                                   shock_step * _get_shock_dist(scaled_time_span, temp, eps_n))
            dz_tmp = dz_tmp - z_step
            num_step = num_step + 1
            if is_regular:
                # for regular materials
                if non_linearity and attenuation is False:
                    temp = burgers_solve(scaled_time_span, temp, temp, eps_n, z_step)
                elif non_linearity and attenuation:
                    temp = burgers_solve(scaled_time_span, temp, temp, eps_n, z_step / 2)
                    temp = attenuation_solve(scaled_time_span, temp, z_step, eps_a, eps_b)
                    temp = burgers_solve(scaled_time_span, temp, temp, eps_n, z_step / 2)
                elif non_linearity is False and attenuation:
                    temp = attenuation_solve(scaled_time_span, temp, z_step, eps_a, eps_b)
            else:
                raise NotImplementedError
        _wave_field[:, index] = temp

    if num_dimensions == 3:
        _wave_field = _wave_field.reshape((num_points_t, num_points_y, num_points_x))

    return _wave_field


def _get_shock_dist(time_span,
                    pressure,
                    eps_n):
    """
    Calculates the approximated distance until shock formation for a regular pulse.
    :param time_span: Time span.
    :param pressure: Pressure at time points
    :param eps_n: Coefficient of non-linearity.
    :return:
    """
    resolution_t = numpy.diff(time_span)
    if numpy.min(resolution_t) <= 0.0:
        return 0.0

    # calculate shock distance
    dp_dt_max = numpy.max(numpy.diff(pressure) / resolution_t)
    if dp_dt_max > 0.0 and eps_n != 0.0:
        shock_dist = 1 / (eps_n * dp_dt_max)
    else:
        shock_dist = numpy.inf

    return shock_dist


# TODO remove Matlab form
def _make_banded(
        vector_a: numpy.ndarray,
        vector_d: numpy.ndarray = None,
        row_major: bool = False) -> numpy.ndarray:
    """
    Returning the diagonals of vector_a specified in vector_d on BLAS banded matrix form.
    :param vector_a:  Matrix to be treated. The matrix is assumed to be of
        standard, column major, Matlab form.
    :param vector_d: Diagonals on vector form (-kl:ku) where 0 denotes main
        diagonal, -kl denotes the kl'th sub-diagonal and ku the ku'th super-diagonal
    :param row_major: Used if the matrix is supposed to be in row-major storage (C-style).
    :return: The vector_a on banded form.
    """
    if vector_d is None:
        m, n = vector_a.shape
        _vector_d = numpy.range(-m + 1, n - 1, dtype=int)
    else:
        _vector_d = vector_d

    num_points_x = numpy.min(vector_a.shape)
    banded = numpy.zeros((numpy.max(_vector_d.shape), num_points_x))

    for index in range(_vector_d):
        diag = numpy.diag(vector_a, _vector_d[index])
        idd = numpy.range(numpy.max(diag.shape))
        stidx = numpy.max(numpy.sign(_vector_d[index]) * (num_points_x - numpy.max(diag.shape)),
                          0)
        banded[index, idd + stidx] = diag

    if row_major:
        banded = banded.T

    return banded
