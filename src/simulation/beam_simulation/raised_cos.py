# -*- coding: utf-8 -*-
"""
    raised_cos.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
from typing import Optional

import numpy


def raised_cos(window_length: int,
               tap_length: int,
               total_length: Optional[int] = None) -> numpy.ndarray:
    """
    Returning a Tukey (or raised cosine) window. The window is flat at the middle
    and tapered to zero at the ends with a cosine shape.
    :param window_length: Length of window. Includes both the part equal to one and the
        length of the cosine tapering.
    :param tap_length: Length of tapering region. The length of the window equal to
        one is thus win_length - tap_length.
    :param total_length: The total length of the returned vector. The region
        outside the window is set to zero.
    :return: Window function.
    """
    if total_length is None:
        _total_length = window_length
    else:
        _total_length = total_length

    if _total_length < window_length:
        _total_length = window_length

    # initialize window
    window = numpy.ones(window_length)

    # calculate tapering
    if tap_length > 1:
        x_tap = numpy.arange(0, tap_length) / (tap_length - 1)
        window[:tap_length] = (1 - numpy.cos(numpy.pi * x_tap)) / 2.0
        window[window_length - tap_length:window_length] = numpy.flipud(window[:tap_length])
    else:
        window[0] = 0
        window[-1] = 0

    # calculate zero-pad region
    l1 = int(numpy.floor((_total_length - window_length) / 2))
    l2 = _total_length - l1 - window_length

    # zero-pad window
    if l1 > 0:
        window = numpy.concatenate((numpy.zeros(l1), window))
    if l2 > 0:
        window = numpy.concatenate((window, numpy.zeros(l2)))

    return window
