"""
simulation.py
"""
import time
from typing import Tuple, List

import numpy
import scipy.sparse

from simulation.beam_simulation.adjust_equidistant_steps import adjust_equidistant_steps
from simulation.beam_simulation.calc_spatial_window import calc_spatial_window
from simulation.beam_simulation.find_steps import find_steps
from simulation.beam_simulation.propagate_through_body_wall import propagate_through_body_wall
from simulation.beam_simulation.recalculate_wave_numbers import recalculate_wave_numbers
from simulation.controls.consts import NoHistory
from simulation.controls.consts import ProfileHistory
from simulation.controls.main_control import MainControl
from simulation.estimate_eta import estimate_eta
from simulation.get_wave_numbers import get_wave_numbers
from simulation.post_processing.export_beam_profile import export_beam_profile
from simulation.propagation.propagate import propagate
from simulation.reporting_simulation_type import reporting_simulation_type


def simulation(control: MainControl,
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

    _num_steps, _step_sizes, _step_idx = _calc_steps(control,
                                                     _current_pos,
                                                     _end_point,
                                                     _step_size,
                                                     _store_pos)

    # adjust _equidistant_steps size flag
    _diff_step_idx, _equidistant_steps, _recalculate = \
        adjust_equidistant_steps(control.diffraction_type,
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
    _window = calc_spatial_window(control,
                                  window,
                                  _annular_transducer,
                                  _num_points_x,
                                  _num_points_y,
                                  _resolution_x,
                                  _resolution_y,
                                  _step_size)

    # reporting simulation type
    reporting_simulation_type(_non_linearity,
                              _num_dimensions,
                              _num_points_t,
                              _num_points_x,
                              _num_points_y)

    # calculating beam profiles
    _ax_pulse, _max_profile, _rms_profile, _z_pos = _calc_beam_profiles(control,
                                                                        _history,
                                                                        _num_points_t,
                                                                        _num_points_x,
                                                                        _num_points_y,
                                                                        _num_steps,
                                                                        _step_index,
                                                                        wave_field)

    # Propagating through body wall
    _ax_pulse, _max_profile, _rms_profile, _wave_field, _z_pos = \
        propagate_through_body_wall(control,
                                    phantom,
                                    _wave_numbers,
                                    _ax_pulse,
                                    _current_pos,
                                    _equidistant_steps,
                                    _history,
                                    _max_profile,
                                    _rms_profile,
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
            recalculate_wave_numbers(control,
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
        _rms_profile, _max_profile, _ax_pulse, _z_pos = export_beam_profile(control,
                                                                            _wave_field,
                                                                            _rms_profile,
                                                                            _max_profile,
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

    return _wave_field, _rms_profile, _max_profile, _ax_pulse, _z_pos


def _calc_steps(control, _current_pos, _end_point, _step_size, _store_pos) \
        -> Tuple[int, List[float], List[int]]:
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
    return _num_steps, _step_sizes, _step_idx,


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


def _calc_beam_profiles(control,
                        history,
                        num_points_t,
                        num_points_x,
                        num_points_y,
                        num_steps,
                        step_index,
                        wave_field):
    if history != NoHistory:
        _rms_profile = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        _max_profile = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        _ax_pulse = numpy.zeros((num_points_t, num_steps))
        _z_pos = numpy.zeros(num_steps)
    else:
        _rms_profile = numpy.array([])
        _max_profile = numpy.array([])
        _ax_pulse = numpy.array([])
        _z_pos = numpy.array([])

    _step_index = 0
    _rms_profile, _max_profile, _ax_pulse, _z_pos = export_beam_profile(control,
                                                                        wave_field,
                                                                        _rms_profile,
                                                                        _max_profile,
                                                                        _ax_pulse,
                                                                        _z_pos,
                                                                        step_index)

    return _ax_pulse, _max_profile, _rms_profile, _z_pos
