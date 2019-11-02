"""
log2_round_off
"""

import math


def log2_round_off(input_value):
    """
    Function finding the nearest number who's factors are only 2. One of the
    factors may be three. Any number less than ten percent downward is ok,
    the rest is rounded upward.
    """
    if 0.0 <= input_value <= 2.0:
        return input_value

    _exponent = math.ceil(math.log2(input_value))
    _factored_value = 2 ** _exponent

    _temp = 2 ** (_exponent - 1)
    if input_value / _temp < 1.1:
        _factored_value = _temp
    else:
        _temp = 2 ** (_exponent - 2) * 3
        if input_value < _temp:
            _factored_value = _temp

    return int(_factored_value)
