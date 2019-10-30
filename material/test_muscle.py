# pylint: disable-all

import unittest

from material.muscle import Muscle


class TestMuscle(unittest.TestCase):
    def setUp(self):
        self.muscle = Muscle(37.0)

    def test_wave_speed(self):
        self.assertAlmostEqual(1549.9, self.muscle.sound_speed)

    def test_mass_density(self):
        self.assertAlmostEqual(1060.0, self.muscle.mass_density)

    def test_non_linearity_coefficient(self):
        self.assertAlmostEqual(3.9, self.muscle.non_linearity_coefficient)

    def test_constant_of_attenuation(self):
        self.assertAlmostEqual(0.52, self.muscle.constant_of_attenuation)

    def test_exponent_of_attenuation(self):
        self.assertAlmostEqual(1.1, self.muscle.exponent_of_attenuation)

    def test_is_regular(self):
        self.assertTrue(self.muscle.is_regular)
