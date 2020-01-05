"""
raised_cos.py

Copyright (C) 2020  Jaeho Kim

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
    _window = numpy.ones(window_length)

    # calculate tapering
    if tap_length > 1:
        _xtap = numpy.arange(0, tap_length) / (tap_length - 1)
        _window[:tap_length] = (1 - numpy.cos(numpy.pi * _xtap)) / 2.0
        _window[window_length - tap_length:window_length] = numpy.flipud(_window[:tap_length])
    else:
        _window[0] = 0
        _window[-1] = 0

    # calculate zero-pad region
    _l1 = int(numpy.floor((_total_length - window_length) / 2))
    _l2 = _total_length - _l1 - window_length

    # zero-pad window
    if _l1 > 0:
        _window = numpy.concatenate((numpy.zeros(_l1), _window))
    if _l2 > 0:
        _window = numpy.concatenate((_window, numpy.zeros(_l2)))

    return _window
