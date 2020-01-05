"""
test_diffraction.py

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

from system.diffraction.diffraction import NoDiffraction, \
    ExactDiffraction, \
    AngularSpectrumDiffraction, \
    PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, \
    FiniteDifferenceTimeDifferenceFull


class DiffractionComparison(unittest.TestCase):
    def test_SameTypes(self):
        self.assertEqual(NoDiffraction, NoDiffraction)

    def test_SameTypeWithOperator(self):
        self.assertTrue(NoDiffraction == NoDiffraction)

    def test_DifferentTypes(self):
        self.assertNotEqual(NoDiffraction, ExactDiffraction)

    def test_DifferentTypesWithOperator(self):
        self.assertTrue(NoDiffraction != ExactDiffraction)

    def test_CompareAllTypes(self):
        self.assertTrue(NoDiffraction !=
                        ExactDiffraction !=
                        AngularSpectrumDiffraction !=
                        PseudoDifferential !=
                        FiniteDifferenceTimeDifferenceReduced !=
                        FiniteDifferenceTimeDifferenceFull)

    def test_membership_true(self):
        self.assertTrue(
            NoDiffraction in (NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction))

    def test_membership_false(self):
        self.assertFalse(NoDiffraction in (ExactDiffraction, AngularSpectrumDiffraction))
