"""
get_window.py
"""
import numpy

from misc.raised_cos import raised_cos


def get_window(num_points,
               resolution,
               _window_length,
               _window_length0,
               annular_transducer: bool) -> numpy.ndarray:
    """
    Function returning spatial window to avoid boundary reflections.
    :param num_points: One- or two-element vector with the number of points in
            (x,y)-direction. If length(N)=1, then N(2), or ny, is assumed
            to be 1. Default [256 1].
    :param resolution: One- or two-element vector with the resolution in the
            (x,y)-direction. If length(ds)=1, the equal resolution in x
            and y is assumed. Default 1/N(1).
    :param _window_length: One- or two-element vector with the physical length of the
            fall-off region of the window in the (x,y)-direction. If
            length(L)=1, then the same length in x and y is assumed.
            Default is 0.1.
    :param _window_length0: One- or two-element vector with the physical length of the
            zero part region of the window in the (x,y)-direction. If
            length(L)=1, then the same length in x and y is assumed.
            Default is 0.1.
    :param annular_transducer: Flag for annular transducer. Returns half-window for 2D
            axi-symmetric simulations.
    :return: Window in a vector ready to be multiplied by ones(1,nt) and applied to wave field
    """
    if isinstance(num_points, int):
        num_points = (num_points, 1)
    if num_points[0] == 1 and num_points[1] == 1:
        raise ValueError('Unsupported Window')

    # create two element vectors
    if isinstance(resolution, float):
        resolution = (resolution, resolution)
    if isinstance(_window_length, float):
        _window_length = (_window_length, _window_length)
    if isinstance(_window_length0, float):
        _window_length0 = (_window_length0, _window_length0)

    # calculate number of points of fall-off and zero part
    _num_points_of_fall_off_x = int(numpy.ceil(_window_length[0] / resolution[0]))
    _num_points_of_zero_x = int(numpy.ceil(_window_length0[0] / resolution[0]))

    # adjust _total_num_x for annular_transducer = 1
    _total_num_y = num_points[1]
    if _total_num_y == 1 and annular_transducer:
        _total_num_x = 2 * num_points[0]
        _num_points_of_fall_off_y = 0
        _num_points_of_zero_y = 0
    else:
        _total_num_x = num_points[0]
        _num_points_of_fall_off_y = int(numpy.ceil(_window_length[1] / resolution[1]))
        _num_points_of_zero_y = int(numpy.ceil(_window_length0[1] / resolution[1]))

    # find length of window
    _window_length_x = _total_num_x - 2 * _num_points_of_zero_x
    _window_length_y = _total_num_y - 2 * _num_points_of_zero_y

    # create window in x
    _wx = raised_cos(_window_length_x, _num_points_of_fall_off_x, _total_num_x)
    if _total_num_y == 1 and annular_transducer:
        _wx = _wx[num_points[0]:]
        _total_num_x = _total_num_x / 2

    # create window in y
    if _total_num_y == 1:
        _wy = numpy.array([1])
    else:
        _wy = raised_cos(_window_length_y, _num_points_of_fall_off_y, _total_num_y)

    # reshape window
    _window = _wy[..., numpy.newaxis] * numpy.transpose(_wx)
    _window = _window.reshape(_total_num_x * _total_num_y)

    return _window
