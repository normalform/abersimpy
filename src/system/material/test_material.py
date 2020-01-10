# -*- coding: utf-8 -*-
"""
    test_material.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
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
