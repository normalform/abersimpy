"""
get_string_position.py
TODO: Need better implementation in Python's way
"""
import numpy


def get_string_position(position: float):
    """
    Function that returns a position as a string. Input is given in mm and
    the string containing 5 digits is organized such that the last digit
    represents a 1/100 of a millimeter. Any length up to 99.999 cm may be
    expressed uniquely up to a precision of 0.001 cm.
    :param position: The position.
    :return: Position as a string _XXXXX
    """
    _position = numpy.round(100 * position) / 100
    _position = _position / 100
    _n = 6

    _num = numpy.zeros(_n, dtype=int)
    for _index in range(_n):
        _num[_index] = int(numpy.floor(_position))
        _position = _position - _num[_index]
        _position = 10 * _position

    _temp = _num[-1]
    _cl = 0
    for _index in range(_n - 1):
        if _num[_n - _index - 1] == 9 and _temp == 9:
            _cl = _cl + 1
            _temp = _num[_n - _index - 1]
        else:
            _temp = _num[_n - _index - 1]

    if _cl > 1:
        _num[_n - 1] = 0
        for _index in range(_cl):
            _num[_n - _index - 1] = int(numpy.mod(_num[_n - _index - 1] + 1, 10))

    _output = '_{:1d}{:1d}{:1d}{:1d}{:1d}'.format(_num[0], _num[1], _num[2], _num[3], _num[4])

    return _output
