# -*- coding: utf-8 -*-
"""
    test_log2_round_off.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
# pylint: disable-all

import unittest

from simulation.controls.log2_round_off import log2_round_off


class DiffractionComparison(unittest.TestCase):
    def test_case0(self):
        self.assertAlmostEqual(16.0, log2_round_off(16.0))
        self.assertAlmostEqual(16.0, log2_round_off(15.0))
        self.assertAlmostEqual(16.0, log2_round_off(14.0))
        self.assertAlmostEqual(16.0, log2_round_off(13.0))
        self.assertAlmostEqual(16.0, log2_round_off(12.0))
        self.assertAlmostEqual(12.0, log2_round_off(11.0))
        self.assertAlmostEqual(12.0, log2_round_off(10.0))
        self.assertAlmostEqual(12.0, log2_round_off(9.0))
        self.assertAlmostEqual(8.0, log2_round_off(8.0))
        self.assertAlmostEqual(8.0, log2_round_off(7.0))
        self.assertAlmostEqual(8.0, log2_round_off(6.0))
        self.assertAlmostEqual(6.0, log2_round_off(5.0))
        self.assertAlmostEqual(4.0, log2_round_off(4.0))
        self.assertAlmostEqual(4.0, log2_round_off(3.0))
