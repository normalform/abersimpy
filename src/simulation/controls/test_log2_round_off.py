"""
test_log2_round_off.py

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
