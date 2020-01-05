"""
test_muscle.py

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

from system.material.muscle import Muscle


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
