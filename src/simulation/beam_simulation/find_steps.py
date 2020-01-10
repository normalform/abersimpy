# -*- coding: utf-8 -*-
"""
    find_steps.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import math
from typing import Tuple, List, Optional

import numpy


def find_steps(start_point: float,
               end_point: float,
               step_size: float,
               store_position: Optional[numpy.ndarray] = numpy.array([]),
               screen_position: Optional[numpy.ndarray] = numpy.array([])) \
        -> Tuple[int, List[float], List[int]]:
    """
    Function that simulates propagation from the transducer to a certain distance.
    The function will depending on the kind of propagation (linear or non-linear)
    return the rms beam profile and the maximum temporal pressure for the fundamental and the 2nd
    harmonic component. The profiles are calculated for each step.
    :param start_point: Start point (depth) of simulation.
    :param end_point: End point of simulation.
    :param step_size: main step size of simulation.
    :param store_position: Position to store pulses
    :param screen_position: Depth coordinate of aberration screens.
    :return: (Number of steps to perform,
              Step size for each step,
              Index indicating whether the step is a main step (equidistant stepping may be used)
                or a special step, i.e. a step with step_size different from the main step size)
    """
    # set tolerance and initiate variables
    TOLERANCE = 1e-14
    spec_points = numpy.unique(numpy.concatenate((store_position, screen_position)))
    current_point = start_point
    shape = int(math.ceil(((end_point - start_point) / step_size) + numpy.max(spec_points.shape)))
    step_sizes = [0.0] * shape
    step_indexes = [0] * shape
    num_steps = 0

    # start loop
    index = 0
    spec_index = numpy.where(spec_points > current_point)[0][0]
    while current_point < end_point:
        if spec_index < numpy.max(spec_points.shape):
            spec_distance = spec_points[spec_index] - current_point
        else:
            spec_distance = 100.0
        if spec_distance < step_size:
            if numpy.abs(spec_distance) < TOLERANCE:
                spec_index = spec_index + 1
                continue
            if numpy.abs(spec_distance - step_size) < TOLERANCE:
                step_sizes[index] = step_size
                step_indexes[index] = 0
            else:
                step_sizes[index] = spec_distance
                step_indexes[index] = -1
                spec_index = spec_index + 1
            index = index + 1
        else:
            if spec_distance > step_size and numpy.abs(spec_distance - step_size) > TOLERANCE:
                _temp_num_steps = numpy.ceil(numpy.sum(numpy.array(step_sizes) / step_size))
                spec_distance = _temp_num_steps * step_size - current_point
                step_indexes[index] = 0
                if spec_distance < TOLERANCE:
                    spec_distance = step_size
                    step_indexes[index] = 0
                step_sizes[index] = spec_distance
            else:
                step_sizes[index] = step_size
                step_indexes[index] = 0
            index = index + 1
        if current_point + step_sizes[index] > end_point:
            if numpy.abs(current_point + step_sizes[index] - end_point) > TOLERANCE:
                if numpy.abs(end_point - current_point) < TOLERANCE:
                    step_sizes = step_sizes[:index]
                    step_indexes = step_sizes[:index]
                    num_steps = index - 2
                    break
                step_sizes[index] = end_point - current_point
                step_indexes[index] = 0
                num_steps = index
                break
        current_point = start_point + numpy.sum(numpy.array(step_sizes) / step_size) * step_size
        num_steps = index

    # adjust size of vectors
    step_sizes = step_sizes[:num_steps]
    step_indexes = step_indexes[:num_steps]

    return num_steps, step_sizes, step_indexes
