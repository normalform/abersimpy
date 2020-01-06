"""
test_get_window.py

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
# pylint: disable-all

import unittest

from simulation.beam_simulation.get_window import get_window


# TODO Continue
class TestGetWindow(unittest.TestCase):
    def test_get_window_case1(self):
        num_points_x = 5
        resolution_x = 0.1
        window_length = 0.1
        window_length0 = 0.1
        _window = get_window(num_points_x,
                             resolution_x,
                             window_length,
                             window_length0,
                             False)

    def test_get_window_case2(self):
        num_points_x = 5
        num_points_y = 3
        resolution_x = 0.1
        resolution_y = 0.1
        window_length = 0.1
        window_length0 = 0.1
        _window = get_window((num_points_x, num_points_y),
                             (resolution_x, resolution_y),
                             window_length,
                             window_length0,
                             False)
