# -*- coding: utf-8 -*-
"""
    log2_round_off.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
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

    exponent = math.ceil(math.log2(input_value))
    factored_value = 2 ** exponent

    temp = 2 ** (exponent - 1)
    if input_value / temp < 1.1:
        factored_value = temp
    else:
        temp = 2 ** (exponent - 2) * 3
        if input_value < temp:
            factored_value = temp

    return int(factored_value)
