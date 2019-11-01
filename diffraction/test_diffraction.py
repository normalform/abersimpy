# pylint: disable-all

import unittest

from diffraction.diffraction import NoDiffraction, \
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
        self.assertTrue(NoDiffraction in (NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction))

    def test_membership_false(self):
        self.assertFalse(NoDiffraction in (ExactDiffraction, AngularSpectrumDiffraction))
