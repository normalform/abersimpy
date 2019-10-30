# pylint: disable-all

import unittest

from prop_control import NoDiffraction, \
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
