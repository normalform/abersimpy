"""
beam_simulation.py
"""
import time

import numpy
import scipy.sparse

from controls.consts import NoHistory, ProfileHistory
from controls.main_control import MainControl
from diffraction.diffraction import ExactDiffraction, PseudoDifferential
from misc.estimate_eta import estimate_eta
from misc.find_steps import find_steps
from misc.get_window import get_window
from postprocessing.export_beamprofile import export_beamprofile
from propagation.get_wave_numbers import get_wave_numbers
from propagation.propagate import propagate
from simscript.body_wall import body_wall


def beam_simulation(control: MainControl,
                    wave_field: numpy.ndarray,
                    screen=numpy.array([]),
                    window=None,
                    phantom=None):
    """
    Function that simulates propagation from the transducer to a certain distance.
    The function will depending on the kind of propagation (linear or non-linear)
    return the rms beam profile and the maximum temporal pressure for the fundamental
    and the 2nd harmonic component. The profiles are calculated for each step.
    :param control: Controls for simulation.
    :param wave_field: Initial wave field at transducer.
    :param screen: Aberration screen used for correction of transmit ultrasound beam
    :param window: Window for tapering the solution to zero close to the boundaries.
        If _window is a scalar, a _window of with a zero region of length 2 * step_size and a raised
        cosine tapering og length _window * step_size will be used. By default, _window=2.
        If _window is set to -1, no windowing will be performed.
        Overrides whatever specified in control.simulation.num_windows.
    :param phantom: Phantom is a function handle to a function who's input is a (x,y,z) coordinate
        and output is the local wave speed at the coordinate.
        The phantom is initialized by a function call with one string argument and finalized with
        a function call with no input arguments. Initialization is performed by the function.
    :return: Wave field at the end of the simulation,
             Temporal RMS beam profile for all frequencies.
                The profile has dimensions (num_points_y * num_points_x * num_steps * harmonic),
                possibly with num_points_y as singleton dimension. The index of the
                harmonic-dimension gives the n'th harmonic profile starting with 0 as the
                total field.
             Maximum pressure in the temporal direction. The profile is structured as the
                RMS profile. The maximum is found as the maximum of the envelope of the signal
                at each point in space.
             Signal located at the control.transducer.center_channel index in space.
                This is the raw signal without any filtering performed for each step.
             The z-coordinate of each profile and axial pulse.
    """
    # calculate number of propagation steps beyond the body wall
    _current_pos = control.simulation.current_position
    _end_point = control.simulation.endpoint
    _step_size = control.simulation.step_size
    _store_pos = control.simulation.store_position
    _equidistant_steps = control.equidistant_steps

    _num_steps, _step_idx, _step_sizes = _calc_steps(control,
                                                     _current_pos,
                                                     _end_point,
                                                     _step_size,
                                                     _store_pos)

    # adjust _equidistant_steps size flag
    _diff_step_idx, _equidistant_steps, _recalculate = _adjust_equidistant_steps(control,
                                                                                 _equidistant_steps,
                                                                                 _num_steps,
                                                                                 _step_idx)

    # sets sizes
    _num_points_x = control.domain.num_points_x
    _num_points_y = control.domain.num_points_y
    _num_points_t = control.domain.num_points_t
    _non_linearity = control.non_linearity
    _annular_transducer = control.annular_transducer
    _history = control.history
    _num_dimensions = control.num_dimensions
    _resolution_x = control.signal.resolution_x
    _resolution_y = control.signal.resolution_y

    # initializing variables
    if screen.size != 0:
        raise NotImplementedError
    _file_name = control.simulation_name

    _wave_numbers = get_wave_numbers(control, _equidistant_steps)

    _times_for_eta = [0.0] * (_num_steps + 1)
    _lap_time_for_eta = 0.0
    _step_index = 0

    # calculate spatial _window
    _window = _calc_spatial_window(control,
                                   window,
                                   _annular_transducer,
                                   _num_points_x,
                                   _num_points_y,
                                   _resolution_x,
                                   _resolution_y,
                                   _step_size)

    # reporting simulation type
    _reporting_simulation_type(_non_linearity,
                               _num_dimensions,
                               _num_points_t,
                               _num_points_x,
                               _num_points_y)

    # calculating beam profiles
    _ax_pulse, _max_pro, _rms_pro, _z_pos = _calc_beam_profiles(control,
                                                                _history,
                                                                _num_points_t,
                                                                _num_points_x,
                                                                _num_points_y,
                                                                _num_steps,
                                                                _step_index,
                                                                wave_field)

    # Propagating through body wall
    _ax_pulse, _max_pro, _rms_pro, _wave_field, _z_pos = \
        _propagate_through_body_wall(control,
                                     phantom,
                                     _wave_numbers,
                                     _ax_pulse,
                                     _current_pos,
                                     _equidistant_steps,
                                     _history,
                                     _max_pro,
                                     _rms_pro,
                                     wave_field,
                                     _window,
                                     _z_pos)

    # Make window into sparse matrix
    _window = _make_window_into_sparse_matrix(_window)

    # Propagating the rest of the distance
    for _index in range(_num_steps - 1):
        _step_index = _step_index + 1
        _start_time = time.time()

        # recalculate wave number operator
        _wave_numbers, _equidistant_steps = \
            _recalculate_wave_numbers(control,
                                      _wave_numbers,
                                      _diff_step_idx,
                                      _equidistant_steps,
                                      _index,
                                      _recalculate,
                                      _step_idx)

        # Propagation
        control.simulation.step_size = _step_sizes[_index]
        _wave_field = propagate(control,
                                _wave_field,
                                direction=1,
                                equidistant_steps=_equidistant_steps,
                                wave_numbers=_wave_numbers)

        # windowing of solution
        _wave_field = _solution_windowing(_num_dimensions,
                                          _num_points_t,
                                          _num_points_x,
                                          _num_points_y,
                                          _wave_field,
                                          _window)

        # calculate beam profiles
        _rms_pro, _max_pro, _ax_pulse, _z_pos = export_beamprofile(control,
                                                                   _wave_field,
                                                                   _rms_pro,
                                                                   _max_pro,
                                                                   _ax_pulse,
                                                                   _z_pos,
                                                                   _step_index)

        _elapsed_time = time.time() - _start_time
        _times_for_eta[_index + 1] = _times_for_eta[_index] + _elapsed_time
        _lap_time_for_eta = estimate_eta(_times_for_eta, _num_steps, _index, _lap_time_for_eta)

    print('Simulation finished in {:.2f} min using an average of {} sec per step.'
          .format(_times_for_eta[-2] / 60.0, numpy.mean(numpy.diff(_times_for_eta[:-2]))))

    # saving the last profiles
    if _history == ProfileHistory:
        print(f'[DUMMY] Saving the last profiles to {_file_name}.json')

    return _wave_field, _rms_pro, _max_pro, _ax_pulse, _z_pos


def _calc_steps(control, _current_pos, _end_point, _step_size, _store_pos):
    if control.heterogeneous_medium and \
            control.simulation.current_position < control.material.thickness:
        _num_steps, _step_sizes, _step_idx = find_steps(control.material.thickness,
                                                        _end_point,
                                                        _step_size,
                                                        _store_pos)
    else:
        _num_steps, _step_sizes, _step_idx = find_steps(_current_pos,
                                                        _end_point,
                                                        _step_size,
                                                        _store_pos)
    return _num_steps, _step_idx, _step_sizes


def _make_window_into_sparse_matrix(_window):
    # TODO check and improve usage of this condition of 'if'
    if _window[0] != -1:
        _nw = numpy.max(_window.shape)
        _window = scipy.sparse.spdiags(_window, 0, _nw, _nw)
    return _window


def _solution_windowing(_num_dimensions, _num_points_t, _num_points_x, _num_points_y, _wave_field,
                        _window):
    if _window.data[0, 0] != -1:
        # TODO check and improve usage of this condition of 'if'
        if _num_dimensions == 3:
            _wave_field = _wave_field.reshape((_num_points_t, _num_points_x * _num_points_y))
        _wave_field = _wave_field * _window
        if _num_dimensions == 3:
            _wave_field = _wave_field.reshape((_num_points_t, _num_points_y, _num_points_x))
    return _wave_field


def _recalculate_wave_numbers(control,
                              wave_numbers,
                              diff_step_idx,
                              equidistant_steps,
                              index,
                              recalculate,
                              step_idx):
    if recalculate:
        if diff_step_idx[index] != 0:
            if step_idx[index] == 0:
                _equidistant_steps = False
            else:
                _equidistant_steps = True
            _wave_numbers = get_wave_numbers(control, _equidistant_steps)
        else:
            _equidistant_steps = equidistant_steps
            _wave_numbers = wave_numbers
    else:
        _equidistant_steps = equidistant_steps
        _wave_numbers = wave_numbers

    return _wave_numbers, _equidistant_steps


def _propagate_through_body_wall(control,
                                 phantom,
                                 wave_numbers,
                                 ax_pulse,
                                 current_pos,
                                 equidistant_steps,
                                 history,
                                 max_pro,
                                 rms_pro,
                                 wave_field,
                                 window,
                                 z_pos):
    if control.heterogeneous_medium != 0 and current_pos < control.material.thickness:
        print('Entering body wall')
        _wave_field, _rms_pro, _max_pro, _ax_pulse, _z_pos = body_wall(control,
                                                                       wave_field,
                                                                       1,
                                                                       equidistant_steps,
                                                                       wave_numbers,
                                                                       window,
                                                                       phantom)
        print('Done with body wall')

        if history != NoHistory:
            _pnx, _pny, _pns, _pnh = _rms_pro.shape
            _step_index = _pns
            raise NotImplementedError
    else:
        _ax_pulse = ax_pulse
        _max_pro = max_pro
        _rms_pro = rms_pro
        _wave_field = wave_field
        _z_pos = z_pos

    return _ax_pulse, _max_pro, _rms_pro, _wave_field, _z_pos


def _calc_beam_profiles(control,
                        history,
                        num_points_t,
                        num_points_x,
                        num_points_y,
                        num_steps,
                        step_index,
                        wave_field):
    if history != NoHistory:
        _rms_pro = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        _max_pro = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        _ax_pulse = numpy.zeros((num_points_t, num_steps))
        _z_pos = numpy.zeros(num_steps)
    else:
        _rms_pro = numpy.array([])
        _max_pro = numpy.array([])
        _ax_pulse = numpy.array([])
        _z_pos = numpy.array([])

    _step_index = 0

    _rms_pro, _max_pro, _ax_pulse, _z_pos = export_beamprofile(control,
                                                               wave_field,
                                                               _rms_pro,
                                                               _max_pro,
                                                               _ax_pulse,
                                                               _z_pos,
                                                               step_index)

    return _ax_pulse, _max_pro, _rms_pro, _z_pos


def _reporting_simulation_type(non_linearity,
                               num_dimensions,
                               num_points_t,
                               num_points_x,
                               num_points_y):
    if non_linearity:
        _non_linearity_str = 'non-linear'
    else:
        _non_linearity_str = 'linear'
    if num_dimensions == 2:
        _dimensions_str = f'{num_points_x}X * {num_points_t}T'
    else:
        _dimensions_str = f'{num_points_x}X * {num_points_y}Y * {num_points_t}T'

    print(f'Starting {_non_linearity_str} simulation of size {_dimensions_str}')


def _calc_spatial_window(control,
                         window,
                         annular_transducer,
                         num_points_x,
                         num_points_y,
                         resolution_x,
                         resolution_y,
                         step_size):
    if window is None:
        _window = control.simulation.num_windows
    else:
        _window = window
    if isinstance(_window, int) and _window > 0:
        _window = get_window((num_points_x, num_points_y),
                             (resolution_x, resolution_y),
                             _window * step_size,
                             2 * step_size,
                             annular_transducer)

    return _window


def _adjust_equidistant_steps(control,
                              equidistant_steps,
                              num_steps,
                              step_idx):
    _diff_step_idx = numpy.concatenate((numpy.array([0], dtype=int), numpy.diff(step_idx)))
    _num_recalculate = len(numpy.where(_diff_step_idx)[0])
    if control.diffraction_type == ExactDiffraction or \
            control.diffraction_type == PseudoDifferential:
        if control.diffraction_type == ExactDiffraction:
            _diffraction_factor = 1
        else:
            _diffraction_factor = 3
        if _num_recalculate / num_steps < 0.5 / _diffraction_factor:
            _recalculate = True
            if step_idx[0] == 0:
                _equidistant_steps = True
            else:
                _equidistant_steps = equidistant_steps
        else:
            _recalculate = False
            _equidistant_steps = False
    else:
        _recalculate = False
        _equidistant_steps = equidistant_steps

    return _diff_step_idx, _equidistant_steps, _recalculate
