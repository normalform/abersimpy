# -*- coding: utf-8 -*-
"""
    test_adjust_equidistant_steps.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
# pylint: disable-all

import unittest

from simulation.beam_simulation.adjust_equidistant_steps import adjust_equidistant_steps
from simulation.beam_simulation.adjust_equidistant_steps import differences
from simulation.beam_simulation.adjust_equidistant_steps import get_num_recalculate
from system.diffraction.diffraction import ExactDiffraction, PseudoDifferential, \
    AngularSpectrumDiffraction


class TestAdjustEquidistantSteps(unittest.TestCase):
    def test_differences(self):
        sequence = [1, 2, 3, 4, 6]
        self.assertListEqual([1, 1, 1, 2], list(differences(sequence)))

    def test_get_num_recalculateCase1(self):
        sequence = [0, 0, 0, 0, 0]
        self.assertEqual(0, get_num_recalculate(sequence))

    def test_get_num_recalculateCase2(self):
        sequence = [0, 0, 1, 2, 2]
        self.assertEqual(3, get_num_recalculate(sequence))

    def test_WithExactDiffractionCase1(self):
        num_steps = 4
        step_index = [0] * num_steps
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     True,
                                     num_steps,
                                     step_index)

        self.assertTrue(equidistant_steps)
        self.assertTrue(recalculate)
        self.assertListEqual(step_index, diff_step_idx)

    def test_WithExactDiffractionCase2(self):
        num_steps = 4
        step_index = [0] * num_steps
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertTrue(equidistant_steps)
        self.assertTrue(recalculate)
        self.assertListEqual(step_index, diff_step_idx)

    def test_WithExactDiffractionCase3(self):
        num_steps = 4
        step_index = list(range(num_steps))
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     True,
                                     num_steps,
                                     step_index)

        self.assertFalse(equidistant_steps)
        self.assertFalse(recalculate)
        self.assertListEqual([0, 1, 1, 1], diff_step_idx)

    def test_WithExactDiffractionCase4(self):
        num_steps = 4
        step_index = [0, 0, 0, 1]
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertTrue(equidistant_steps)
        self.assertTrue(recalculate)
        self.assertListEqual([0, 0, 0, 1], diff_step_idx)

    def test_WithExactDiffractionCase5(self):
        num_steps = 4
        step_index = [1, 2, 2, 2]
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertFalse(equidistant_steps)
        self.assertTrue(recalculate)
        self.assertListEqual([0, 1, 0, 0], diff_step_idx)

    def test_WithPseudoDifferentialCase1(self):
        num_steps = 4
        step_index = [0] * num_steps
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     True,
                                     num_steps,
                                     step_index)

        self.assertTrue(equidistant_steps)
        self.assertTrue(recalculate)
        self.assertListEqual(step_index, diff_step_idx)

    def test_WithPseudoDifferentialCase2(self):
        num_steps = 4
        step_index = [0] * num_steps
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertTrue(equidistant_steps)
        self.assertTrue(recalculate)
        self.assertListEqual(step_index, diff_step_idx)

    def test_WithPseudoDifferentialCase3(self):
        num_steps = 4
        step_index = list(range(num_steps))
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     True,
                                     num_steps,
                                     step_index)

        self.assertFalse(equidistant_steps)
        self.assertFalse(recalculate)
        self.assertListEqual([0, 1, 1, 1], diff_step_idx)

    def test_WithPseudoDifferentialCase4(self):
        num_steps = 4
        step_index = [0, 0, 0, 1]
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertFalse(equidistant_steps)
        self.assertFalse(recalculate)
        self.assertListEqual([0, 0, 0, 1], diff_step_idx)

    def test_WithPseudoDifferentialCase5(self):
        num_steps = 4
        step_index = [1, 2, 2, 2]
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertFalse(equidistant_steps)
        self.assertFalse(recalculate)
        self.assertListEqual([0, 1, 0, 0], diff_step_idx)

    def test_WithAngularSpectrumDiffractionCase1(self):
        num_steps = 10
        step_index = [0] * num_steps
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(AngularSpectrumDiffraction,
                                     True,
                                     num_steps,
                                     step_index)
        self.assertTrue(equidistant_steps)
        self.assertFalse(recalculate)
        self.assertListEqual(step_index, diff_step_idx)

    def test_WithAngularSpectrumDiffractionCase2(self):
        num_steps = 10
        step_index = [0] * num_steps
        diff_step_idx, equidistant_steps, recalculate = \
            adjust_equidistant_steps(AngularSpectrumDiffraction,
                                     False,
                                     num_steps,
                                     step_index)
        self.assertFalse(equidistant_steps)
        self.assertFalse(recalculate)
        self.assertListEqual(step_index, diff_step_idx)
