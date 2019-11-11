# pylint: disable-all

import unittest

import numpy
import numpy.testing

from controls import consts
from controls.material_control import MaterialControl
from material.aberration_phantom import AberrationPhantom
from material.muscle import Muscle


class MaterialControlTests(unittest.TestCase):
    def test_get_thickness_with_aberrationPhantom(self):
        material = AberrationPhantom(37.0)
        self.assertEqual(0.035, MaterialControl._get_thickness(material))

    def test_get_thickness_without_aberrationPhantom(self):
        material = Muscle(37.0)
        self.assertEqual(0.02, MaterialControl._get_thickness(material))

    def test_get_offset(self):
        self.assertEqual([0.0, 0.0], MaterialControl._get_offset())

    def test_calc_delay_screens_amplitude(self):
        numpy.testing.assert_array_almost_equal(numpy.array([[9e-5], [9e-5]]),
                                                MaterialControl._calc_delay_screens_amplitude(2))

    def test_calc_delay_screens_length(self):
        numpy.testing.assert_array_almost_equal(numpy.array([[0.004, 0.1], [0.004, 0.1]]),
                                                MaterialControl._calc_delay_screens_length(2))

    def test_calc_delay_screens_seed(self):
        numpy.testing.assert_array_almost_equal(numpy.array([1.0, 2.0]),
                                                MaterialControl._calc_delay_screens_seed(2))

    def test_get_num_screen_filename_no_aberration(self):
        self.assertEqual('', MaterialControl._get_num_screen_filename(
            consts.NoAberrationAndHomogeneousMedium))

    def test_get_num_screen_filename_with_aberration_from_file(self):
        self.assertEqual('randseq.json',
                         MaterialControl._get_num_screen_filename(consts.AberrationFromFile))

    def test_get_num_screen_filename_with_aberration_phantom(self):
        self.assertEqual('phantom.json',
                         MaterialControl._get_num_screen_filename(consts.AberrationPhantom))
