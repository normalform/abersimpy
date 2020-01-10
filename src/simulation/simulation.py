# -*- coding: utf-8 -*-
"""
    simulation.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
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
from simulation.controls.consts import NO_HISTORY
from simulation.controls.consts import PROFILE_HISTORY
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
    current_pos = control.simulation.current_position
    end_point = control.simulation.endpoint
    step_size = control.simulation.step_size
    store_pos = control.simulation.store_position
    equidistant_steps = control.equidistant_steps

    num_steps, step_sizes, step_idx = _calc_steps(control,
                                                  current_pos,
                                                  end_point,
                                                  step_size,
                                                  store_pos)

    # adjust equidistant_steps size flag
    diff_step_idx, equidistant_steps, recalculate = \
        adjust_equidistant_steps(control.diffraction_type,
                                 equidistant_steps,
                                 num_steps,
                                 step_idx)

    # sets sizes
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t
    non_linearity = control.non_linearity
    annular_transducer = control.annular_transducer
    history = control.history
    num_dimensions = control.num_dimensions
    resolution_x = control.signal.resolution_x
    resolution_y = control.signal.resolution_y

    # initializing variables
    if screen.size != 0:
        raise NotImplementedError
    file_name = control.simulation_name

    wave_numbers = get_wave_numbers(control, equidistant_steps)

    times_for_eta = [0.0] * (num_steps + 1)
    lap_time_for_eta = 0.0
    step_index = 0

    # calculate spatial _window
    _window = calc_spatial_window(control,
                                  window,
                                  annular_transducer,
                                  num_points_x,
                                  num_points_y,
                                  resolution_x,
                                  resolution_y,
                                  step_size)

    # reporting simulation type
    reporting_simulation_type(non_linearity,
                              num_dimensions,
                              num_points_t,
                              num_points_x,
                              num_points_y)

    # calculating beam profiles
    ax_pulse, max_profile, rms_profile, z_pos = _calc_beam_profiles(control,
                                                                    history,
                                                                    num_points_t,
                                                                    num_points_x,
                                                                    num_points_y,
                                                                    num_steps,
                                                                    step_index,
                                                                    wave_field)

    # Propagating through body wall
    ax_pulse, max_profile, rms_profile, _wave_field, z_pos = \
        propagate_through_body_wall(control,
                                    phantom,
                                    wave_numbers,
                                    ax_pulse,
                                    current_pos,
                                    equidistant_steps,
                                    history,
                                    max_profile,
                                    rms_profile,
                                    wave_field,
                                    _window,
                                    z_pos)

    # Make window into sparse matrix
    _window = _make_window_into_sparse_matrix(_window)

    # Propagating the rest of the distance
    for index in range(num_steps - 1):
        step_index = step_index + 1
        start_time = time.time()

        # recalculate wave number operator
        wave_numbers, equidistant_steps = \
            recalculate_wave_numbers(control,
                                     wave_numbers,
                                     diff_step_idx,
                                     equidistant_steps,
                                     index,
                                     recalculate,
                                     step_idx)

        # Propagation
        control.simulation.step_size = step_sizes[index]
        _wave_field = propagate(control,
                                _wave_field,
                                direction=1,
                                equidistant_steps=equidistant_steps,
                                wave_numbers=wave_numbers)

        # windowing of solution
        _wave_field = _solution_windowing(num_dimensions,
                                          num_points_t,
                                          num_points_x,
                                          num_points_y,
                                          _wave_field,
                                          _window)

        # calculate beam profiles
        rms_profile, max_profile, ax_pulse, z_pos = export_beam_profile(control,
                                                                        _wave_field,
                                                                        rms_profile,
                                                                        max_profile,
                                                                        ax_pulse,
                                                                        z_pos,
                                                                        step_index)

        elapsed_time = time.time() - start_time
        times_for_eta[index + 1] = times_for_eta[index] + elapsed_time
        lap_time_for_eta = estimate_eta(times_for_eta, num_steps, index, lap_time_for_eta)

    print('Simulation finished in {:.2f} min using an average of {} sec per step.'
          .format(times_for_eta[-2] / 60.0, numpy.mean(numpy.diff(times_for_eta[:-2]))))

    # saving the last profiles
    if history == PROFILE_HISTORY:
        print(f'[DUMMY] Saving the last profiles to {file_name}.json')

    return _wave_field, rms_profile, max_profile, ax_pulse, z_pos


def _calc_steps(control, current_pos, end_point, step_size, store_pos) \
        -> Tuple[int, List[float], List[int]]:
    if control.heterogeneous_medium and \
            control.simulation.current_position < control.material.thickness:
        num_steps, step_sizes, step_idx = find_steps(control.material.thickness,
                                                     end_point,
                                                     step_size,
                                                     store_pos)
    else:
        num_steps, step_sizes, step_idx = find_steps(current_pos,
                                                     end_point,
                                                     step_size,
                                                     store_pos)
    return num_steps, step_sizes, step_idx,


def _make_window_into_sparse_matrix(window):
    # TODO check and improve usage of this condition of 'if'
    if window[0] != -1:
        nw = numpy.max(window.shape)
        window = scipy.sparse.spdiags(window, 0, nw, nw)
    return window


def _solution_windowing(num_dimensions,
                        num_points_t,
                        num_points_x,
                        num_points_y,
                        wave_field,
                        window):
    if window.data[0, 0] != -1:
        # TODO check and improve usage of this condition of 'if'
        if num_dimensions == 3:
            wave_field = wave_field.reshape((num_points_t, num_points_x * num_points_y))
        wave_field = wave_field * window
        if num_dimensions == 3:
            wave_field = wave_field.reshape((num_points_t, num_points_y, num_points_x))
    return wave_field


def _calc_beam_profiles(control,
                        history,
                        num_points_t,
                        num_points_x,
                        num_points_y,
                        num_steps,
                        step_index,
                        wave_field):
    if history != NO_HISTORY:
        rms_profile = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        max_profile = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        ax_pulse = numpy.zeros((num_points_t, num_steps))
        z_pos = numpy.zeros(num_steps)
    else:
        rms_profile = numpy.array([])
        max_profile = numpy.array([])
        ax_pulse = numpy.array([])
        z_pos = numpy.array([])

    step_index = 0
    rms_profile, max_profile, ax_pulse, z_pos = export_beam_profile(control,
                                                                    wave_field,
                                                                    rms_profile,
                                                                    max_profile,
                                                                    ax_pulse,
                                                                    z_pos,
                                                                    step_index)

    return ax_pulse, max_profile, rms_profile, z_pos
