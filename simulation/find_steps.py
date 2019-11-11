"""
find_steps.py
"""
from typing import Tuple

import numpy


def find_steps(start_point: float,
               end_point: float,
               step_size: float,
               store_position: numpy.ndarray = numpy.array([]),
               screen_position: numpy.ndarray = numpy.array([])) \
        -> Tuple[int, numpy.ndarray, numpy.ndarray]:
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
    _Tolerance = 1e-14
    _spec_points = numpy.unique(numpy.concatenate((store_position, screen_position)))
    _current_point = start_point
    _shape = (int(numpy.ceil(((end_point - start_point) / step_size) +
                             numpy.max(_spec_points.shape))),)
    _step_sizes = numpy.zeros(_shape)
    _step_indexes = numpy.zeros(_shape, dtype=int)
    _num_steps = 0

    # start loop
    _index = 0
    _spec_index = numpy.where(_spec_points > _current_point)[0][0]
    while _current_point < end_point:
        if _spec_index < numpy.max(_spec_points.shape):
            _spec_distance = _spec_points[_spec_index] - _current_point
        else:
            _spec_distance = 100.0
        if _spec_distance < step_size:
            if numpy.abs(_spec_distance) < _Tolerance:
                _spec_index = _spec_index + 1
                continue
            if numpy.abs(_spec_distance - step_size) < _Tolerance:
                _step_sizes[_index] = step_size
                _step_indexes[_index] = 0
            else:
                _step_sizes[_index] = _spec_distance
                _step_indexes[_index] = -1
                _spec_index = _spec_index + 1
            _index = _index + 1
        else:
            if _spec_distance > step_size and numpy.abs(_spec_distance - step_size) > _Tolerance:
                _temp_num_steps = numpy.ceil(numpy.sum(_step_sizes / step_size))
                _spec_distance = _temp_num_steps * step_size - _current_point
                _step_indexes[_index] = 0
                if _spec_distance < _Tolerance:
                    _spec_distance = step_size
                    _step_indexes[_index] = 0
                _step_sizes[_index] = _spec_distance
            else:
                _step_sizes[_index] = step_size
                _step_indexes[_index] = 0
            _index = _index + 1
        if _current_point + _step_sizes[_index] > end_point:
            if numpy.abs(_current_point + _step_sizes[_index] - end_point) > _Tolerance:
                if numpy.abs(end_point - _current_point) < _Tolerance:
                    _step_sizes = _step_sizes[:_index]
                    _step_indexes = _step_sizes[:_index]
                    _num_steps = _index - 2
                    break
                _step_sizes[_index] = end_point - _current_point
                _step_indexes[_index] = 0
                _num_steps = _index
                break
        _current_point = start_point + numpy.sum(_step_sizes / step_size) * step_size
        _num_steps = _index

    # adjust size of vectors
    _step_sizes = _step_sizes[:_num_steps]
    _step_indexes = _step_indexes[:_num_steps]

    return _num_steps, _step_sizes, _step_indexes
