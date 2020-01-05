"""
test_material.py

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

import numpy

from system.material.material import BaseMaterial


class StubMaterial(BaseMaterial):
    def __init__(self, temperature: float):
        super().__init__(temperature)

    @property
    def constant_of_attenuation(self):
        return 1.0

    @property
    def exponent_of_attenuation(self) -> float:
        return 1.0

    @property
    def non_linearity_coefficient(self) -> float:
        return 1.0

    @property
    def mass_density(self) -> float:
        return 1.0

    @property
    def sound_speed(self) -> float:
        return 1.0


class TestBaseMaterial(unittest.TestCase):
    def test_eps_a(self):
        stub = StubMaterial(37.0)
        self.assertAlmostEqual(0.018323389, stub.eps_a)

    def test_eps_b(self):
        stub = StubMaterial(37.0)
        self.assertAlmostEqual(1.0, stub.eps_b)

    def test_eps_n(self):
        stub = StubMaterial(37.0)
        self.assertAlmostEqual(10000000000.0, stub.eps_n)

    def test_compressibility(self):
        stub = StubMaterial(37.0)
        self.assertAlmostEqual(1.0, stub.compressibility)

    def test_is_regular(self):
        stub = StubMaterial(37.0)
        self.assertAlmostEqual(True, stub.is_regular)

    def test_interpolate(self):
        stub = StubMaterial(37.0)
        self.assertAlmostEqual(3.5, stub._interpolation(numpy.array([36.0, 38.0]),
                                                        numpy.array([3.0, 4.0])))
        self.assertAlmostEqual(3.0, stub._interpolation(numpy.array([36.0, 37.0, 38.0]),
                                                        numpy.array([2.0, 3.0, 4.0])))
        # extrapolation cases
        self.assertAlmostEqual(5.0, stub._interpolation(numpy.array([35.0, 36.0]),
                                                        numpy.array([3.0, 4.0])))
        self.assertAlmostEqual(6.0, stub._interpolation(numpy.array([33.0, 34.0, 35.0]),
                                                        numpy.array([2.0, 3.0, 4.0])))
