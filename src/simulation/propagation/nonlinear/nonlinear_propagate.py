"""
nonlinear_propagate.py

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy
import scipy.sparse

from simulation.controls.consts import ScaleForSpatialVariablesZ, ScaleForTemporalVariable
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
    _material = control.material.material
    _sound_speed = _material.sound_speed

    # scaling sound speed
    _sound_speed = _sound_speed / ScaleForSpatialVariablesZ * ScaleForTemporalVariable
    # scale sampling to micro seconds
    _resolution_t = control.signal.resolution_t / ScaleForTemporalVariable
    # _resolution_x scaled to centimeter
    _resolution_x = control.signal.resolution_x / ScaleForSpatialVariablesZ
    # _resolution_y scaled to centimeter
    _resolution_y = control.signal.resolution_y / ScaleForSpatialVariablesZ
    # _resolution_z scaled to centimeter
    _resolution_z = control.signal.resolution_z / ScaleForSpatialVariablesZ

    _num_dimensions = control.num_dimensions
    _num_points_x = control.domain.num_points_x
    _num_points_y = control.domain.num_points_y
    _num_points_t = control.domain.num_points_t

    _annular_transducer = control.annular_transducer
    _shock_step = control.simulation.shock_step
    _step_size = control.simulation.step_size
    _perfect_matching_layer_width = control.domain.perfect_matching_layer_width

    _num_sub_steps = int(numpy.ceil((_step_size / ScaleForSpatialVariablesZ) / _resolution_z))
    _resolution_z = (_step_size / ScaleForSpatialVariablesZ) / _num_sub_steps
    _d = (_sound_speed / 2) * (_resolution_t / 2) * _resolution_z
    _t_span = numpy.linspace(_resolution_t,
                             _num_points_t *
                             _resolution_t + _resolution_t,
                             _num_points_t).T

    # assign flags
    _diffraction_type = control.diffraction_type
    _non_linearity = control.non_linearity
    _attenuation = control.attenuation

    # prepare PML and FD matrices
    if _perfect_matching_layer_width > 0:
        _a = _d * _get_difference_matrix(2 * _perfect_matching_layer_width,
                                         _resolution_x,
                                         4)

    if _diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
            _diffraction_type == FiniteDifferenceTimeDifferenceFull:
        if numpy.abs(_wave_numbers[4] - _d) > 1e-12:
            # find difference matrix A
            _ax = _d * _get_difference_matrix(_num_points_x,
                                              _resolution_x,
                                              4,
                                              _annular_transducer)
            _bx = numpy.eye(_num_points_x) + _ax
            _dx = numpy.eye(_num_points_x) - _ax
            _dx_inv = numpy.inv(_dx)

            if _diffraction_type == FiniteDifferenceTimeDifferenceReduced:
                # Make matrices banded
                _bx_banded = _make_banded(_bx, numpy.arange(-2, 2, dtype=int))
                _dx_inv_banded = _make_banded(_dx_inv, numpy.arange(-10, 10, dtype=int))
            elif _diffraction_type == FiniteDifferenceTimeDifferenceFull:
                raise NotImplementedError
    elif _diffraction_type in (NoDiffraction,
                               ExactDiffraction,
                               AngularSpectrumDiffraction,
                               PseudoDifferential):
        control.simulation.step_size = _resolution_z * ScaleForSpatialVariablesZ

    _wave = wave

    # Nonlinear propagation
    for _index in range(_num_sub_steps):
        # diffraction
        if _diffraction_type in (ExactDiffraction,
                                 AngularSpectrumDiffraction,
                                 PseudoDifferential):
            _wave = propagate.propagate(control,
                                        _wave,
                                        2 * direction,
                                        equidistant_steps,
                                        _wave_numbers)
        elif _diffraction_type in (FiniteDifferenceTimeDifferenceReduced,
                                   FiniteDifferenceTimeDifferenceFull):
            raise NotImplementedError

        # perfectly matching layers, absorbing boundaries
        if _perfect_matching_layer_width > 0:
            if control.num_dimensions == 2:
                _wave = _wave.reshape((_num_points_x, 1, _num_points_t))
                raise NotImplementedError

        # Nonlinear and _attenuation
        if _non_linearity or _attenuation:
            _wave = _nonlinear_attenuation_split(_t_span,
                                                 _wave,
                                                 _resolution_z,
                                                 _shock_step,
                                                 _material,
                                                 _non_linearity,
                                                 _attenuation)

    # set _step_size back to normal
    if _diffraction_type in (ExactDiffraction, AngularSpectrumDiffraction):
        control.simulation.step_size = _step_size

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
    if differential_order is 1:
        if num_order is 2:
            _difference_stencil = numpy.array([-1.0, 0.0, 1.0]) / 2.0
        elif num_order is 4:
            _difference_stencil = numpy.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
        else:
            print('Unrecognized numerical order. Using fourth order differencing')
            _difference_stencil = _get_difference_stencil(differential_order, 4)
    elif differential_order is 2:
        if num_order is 2:
            _difference_stencil = numpy.array([-1.0, 2.0, -1.0])
        elif num_order is 4:
            _difference_stencil = numpy.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
        else:
            print('Unrecognized numerical order. Using fourth order differencing')
            _difference_stencil = _get_difference_stencil(differential_order, 4)
    else:
        print('Unrecognized differential order, Using second order differential')
        _difference_stencil = _get_difference_stencil(2, num_order)

    return _difference_stencil


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
    _col = numpy.ones((num_points, 1))
    _d = _get_difference_stencil(order, 2)
    _a = scipy.sparse.spdiags(_col * _d,
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
    _num_dimensions = wave_field.ndim
    if _num_dimensions is 3:
        _num_points_t, _num_points_y, _num_points_x = wave_field.shape
        _wave_field = wave_field.reshape((_num_points_t, _num_points_x * _num_points_y))
    else:
        _num_points_t, _num_points_x = wave_field.shape
        _num_points_y = 1
        _wave_field = wave_field

    # Check material
    _is_regular = material.is_regular
    _eps_n = material.eps_n
    _eps_a = material.eps_a
    _eps_b = material.eps_b

    for _index in range(_num_points_x * _num_points_y):
        _num_step = 0
        _temp = _wave_field[:, _index]
        _dz_tmp = resolution_z
        while _dz_tmp > 0:
            _z_step = numpy.minimum(_dz_tmp,
                                    shock_step * _get_shock_dist(scaled_time_span, _temp, _eps_n))
            _dz_tmp = _dz_tmp - _z_step
            _num_step = _num_step + 1
            if _is_regular:
                # for regular materials
                if non_linearity and attenuation is False:
                    _temp = burgers_solve(scaled_time_span, _temp, _temp, _eps_n, _z_step)
                elif non_linearity and attenuation:
                    _temp = burgers_solve(scaled_time_span, _temp, _temp, _eps_n, _z_step / 2)
                    _temp = attenuation_solve(scaled_time_span, _temp, _z_step, _eps_a, _eps_b)
                    _temp = burgers_solve(scaled_time_span, _temp, _temp, _eps_n, _z_step / 2)
                elif non_linearity is False and attenuation:
                    _temp = attenuation_solve(scaled_time_span, _temp, _z_step, _eps_a, _eps_b)
            else:
                raise NotImplementedError
        _wave_field[:, _index] = _temp

    if _num_dimensions is 3:
        _wave_field = _wave_field.reshape((_num_points_t, _num_points_y, _num_points_x))

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
    _resolution_t = numpy.diff(time_span)
    if numpy.min(_resolution_t) <= 0.0:
        return 0.0

    # calculate shock distance
    _dpdt_max = numpy.max(numpy.diff(pressure) / _resolution_t)
    if _dpdt_max > 0.0 and eps_n != 0.0:
        _shock_dist = 1 / (eps_n * _dpdt_max)
    else:
        _shock_dist = numpy.inf

    return _shock_dist


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
        _m, _n = vector_a.shape
        _vector_d = numpy.range(-_m + 1, _n - 1, dtype=int)
    else:
        _vector_d = vector_d

    _num_points_x = numpy.min(vector_a.shape)
    _banded = numpy.zeros((numpy.max(_vector_d.shape), _num_points_x))

    for _index in range(_vector_d):
        _diag = numpy.diag(vector_a, _vector_d[_index])
        _idd = numpy.range(numpy.max(_diag.shape))
        _stidx = numpy.max(numpy.sign(_vector_d[_index]) * (_num_points_x - numpy.max(_diag.shape)),
                           0)
        _banded[_index, _idd + _stidx] = _diag

    if row_major:
        _banded = _banded.T

    return _banded
