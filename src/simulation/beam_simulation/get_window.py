"""
get_window.py
"""
from typing import Union, Tuple

import numpy

from simulation.beam_simulation.raised_cos import raised_cos


def get_window(num_points: Union[int, Tuple[int, int]],
               resolution: Union[float, Tuple[float, float]],
               window_length: Union[float, Tuple[float, float]],
               window_length0: Union[float, Tuple[float, float]],
               annular_transducer: bool) -> numpy.ndarray:
    """
    Function returning spatial window to avoid boundary reflections.
    :param num_points: One- or two-element vector with the number of points in
            (x,y)-direction. If len(num_points) is 1, then num_points[1], or ny, is assumed
            to be 1.
    :param resolution: One- or two-element vector with the resolution in the
            (x,y)-direction. If len(resolution) is 1, the equal resolution in x
            and y is assumed.
    :param window_length: One- or two-element vector with the physical length of the
            fall-off region of the window in the (x,y)-direction. If
            len(window_length) is 1, then the same length in x and y is assumed.
    :param window_length0: One- or two-element vector with the physical length of the
            zero part region of the window in the (x,y)-direction. If
            len(window_length0) is 1, then the same length in x and y is assumed.
    :param annular_transducer: Flag for annular transducer. Returns half-window for 2D
            axi-symmetric simulations.
    :return: Window in a vector ready to be multiplied by ones((0,nt)) and applied to wave field
    """
    if isinstance(num_points, int):
        num_points = (num_points, 1)
    if num_points[0] == 1 and num_points[1] == 1:
        raise ValueError('Unsupported Window')

    # create two element vectors
    if isinstance(resolution, float):
        _resolution = (resolution, resolution)
    else:
        _resolution = resolution

    if isinstance(window_length, float):
        _window_length = (window_length, window_length)
    else:
        _window_length = window_length

    if isinstance(window_length0, float):
        _window_length0 = (window_length0, window_length0)
    else:
        _window_length0 = window_length0

    # calculate number of points of fall-off and zero part
    _num_points_of_fall_off_x = int(numpy.ceil(_window_length[0] / _resolution[0]))
    _num_points_of_zero_x = int(numpy.ceil(_window_length0[0] / _resolution[0]))

    # adjust _total_num_x for annular_transducer = 1
    _total_num_y = num_points[1]
    if _total_num_y == 1 and annular_transducer:
        _total_num_x = 2 * num_points[0]
        _num_points_of_fall_off_y = 0
        _num_points_of_zero_y = 0
    else:
        _total_num_x = num_points[0]
        _num_points_of_fall_off_y = int(numpy.ceil(_window_length[1] / _resolution[1]))
        _num_points_of_zero_y = int(numpy.ceil(_window_length0[1] / _resolution[1]))

    # find length of window
    _window_length_x = _total_num_x - 2 * _num_points_of_zero_x
    _window_length_y = _total_num_y - 2 * _num_points_of_zero_y

    # create window in x
    _win_length = int(numpy.floor(_window_length_x))
    _wx = raised_cos(_win_length, _num_points_of_fall_off_x, _total_num_x)
    if _total_num_y == 1 and annular_transducer:
        _wx = _wx[num_points[0]:]
        _total_num_x = _total_num_x / 2

    # create window in y
    if _total_num_y == 1:
        _wy = numpy.array([1])
    else:
        _win_length = int(numpy.floor(_window_length_y))
        _wy = raised_cos(_win_length, _num_points_of_fall_off_y, _total_num_y)

    # reshape window
    _window = _wy[..., numpy.newaxis] * _wx.T
    _window = _window.reshape(_total_num_x * _total_num_y)

    return _window
