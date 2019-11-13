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
