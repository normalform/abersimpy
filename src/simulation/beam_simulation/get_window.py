# -*- coding: utf-8 -*-
"""
    get_window.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
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
    num_points_of_fall_off_x = int(numpy.ceil(_window_length[0] / _resolution[0]))
    num_points_of_zero_x = int(numpy.ceil(_window_length0[0] / _resolution[0]))

    # adjust total_num_x for annular_transducer = 1
    total_num_y = num_points[1]
    if total_num_y == 1 and annular_transducer:
        total_num_x = 2 * num_points[0]
        num_points_of_fall_off_y = 0
        num_points_of_zero_y = 0
    else:
        total_num_x = num_points[0]
        num_points_of_fall_off_y = int(numpy.ceil(_window_length[1] / _resolution[1]))
        num_points_of_zero_y = int(numpy.ceil(_window_length0[1] / _resolution[1]))

    # find length of window
    window_length_x = total_num_x - 2 * num_points_of_zero_x
    window_length_y = total_num_y - 2 * num_points_of_zero_y

    # create window in x
    win_length = int(numpy.floor(window_length_x))
    wx = raised_cos(win_length, num_points_of_fall_off_x, total_num_x)
    if total_num_y == 1 and annular_transducer:
        wx = wx[num_points[0]:]
        total_num_x = total_num_x / 2

    # create window in y
    if total_num_y == 1:
        wy = numpy.array([1])
    else:
        win_length = int(numpy.floor(window_length_y))
        wy = raised_cos(win_length, num_points_of_fall_off_y, total_num_y)

    # reshape window
    window = wy[..., numpy.newaxis] * wx.T
    window = window.reshape(total_num_x * total_num_y)

    return window
