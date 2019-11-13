# pylint: disable-all

import unittest

from simulation.beam_simulation.adjust_equidistant_steps import _differences
from simulation.beam_simulation.adjust_equidistant_steps import _get_num_recalculate
from simulation.beam_simulation.adjust_equidistant_steps import adjust_equidistant_steps
from system.diffraction.diffraction import ExactDiffraction, PseudoDifferential, \
    AngularSpectrumDiffraction


class TestAdjustEquidistantSteps(unittest.TestCase):
    def test_differences(self):
        sequence = [1, 2, 3, 4, 6]
        self.assertListEqual([1, 1, 1, 2], list(_differences(sequence)))

    def test_get_num_recalculateCase1(self):
        sequence = [0, 0, 0, 0, 0]
        self.assertEqual(0, _get_num_recalculate(sequence))

    def test_get_num_recalculateCase2(self):
        sequence = [0, 0, 1, 2, 2]
        self.assertEqual(3, _get_num_recalculate(sequence))

    def test_WithExactDiffractionCase1(self):
        num_steps = 4
        step_index = [0] * num_steps
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     True,
                                     num_steps,
                                     step_index)

        self.assertTrue(_equidistant_steps)
        self.assertTrue(_recalculate)
        self.assertListEqual(step_index, _diff_step_idx)

    def test_WithExactDiffractionCase2(self):
        num_steps = 4
        step_index = [0] * num_steps
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertTrue(_equidistant_steps)
        self.assertTrue(_recalculate)
        self.assertListEqual(step_index, _diff_step_idx)

    def test_WithExactDiffractionCase3(self):
        num_steps = 4
        step_index = list(range(num_steps))
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     True,
                                     num_steps,
                                     step_index)

        self.assertFalse(_equidistant_steps)
        self.assertFalse(_recalculate)
        self.assertListEqual([0, 1, 1, 1], _diff_step_idx)

    def test_WithExactDiffractionCase4(self):
        num_steps = 4
        step_index = [0, 0, 0, 1]
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertTrue(_equidistant_steps)
        self.assertTrue(_recalculate)
        self.assertListEqual([0, 0, 0, 1], _diff_step_idx)

    def test_WithExactDiffractionCase5(self):
        num_steps = 4
        step_index = [1, 2, 2, 2]
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(ExactDiffraction,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertFalse(_equidistant_steps)
        self.assertTrue(_recalculate)
        self.assertListEqual([0, 1, 0, 0], _diff_step_idx)

    def test_WithPseudoDifferentialCase1(self):
        num_steps = 4
        step_index = [0] * num_steps
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     True,
                                     num_steps,
                                     step_index)

        self.assertTrue(_equidistant_steps)
        self.assertTrue(_recalculate)
        self.assertListEqual(step_index, _diff_step_idx)

    def test_WithPseudoDifferentialCase2(self):
        num_steps = 4
        step_index = [0] * num_steps
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertTrue(_equidistant_steps)
        self.assertTrue(_recalculate)
        self.assertListEqual(step_index, _diff_step_idx)

    def test_WithPseudoDifferentialCase3(self):
        num_steps = 4
        step_index = list(range(num_steps))
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     True,
                                     num_steps,
                                     step_index)

        self.assertFalse(_equidistant_steps)
        self.assertFalse(_recalculate)
        self.assertListEqual([0, 1, 1, 1], _diff_step_idx)

    def test_WithPseudoDifferentialCase4(self):
        num_steps = 4
        step_index = [0, 0, 0, 1]
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertFalse(_equidistant_steps)
        self.assertFalse(_recalculate)
        self.assertListEqual([0, 0, 0, 1], _diff_step_idx)

    def test_WithPseudoDifferentialCase5(self):
        num_steps = 4
        step_index = [1, 2, 2, 2]
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(PseudoDifferential,
                                     False,
                                     num_steps,
                                     step_index)

        self.assertFalse(_equidistant_steps)
        self.assertFalse(_recalculate)
        self.assertListEqual([0, 1, 0, 0], _diff_step_idx)

    def test_WithAngularSpectrumDiffractionCase1(self):
        num_steps = 10
        step_index = [0] * num_steps
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(AngularSpectrumDiffraction,
                                     True,
                                     num_steps,
                                     step_index)
        self.assertTrue(_equidistant_steps)
        self.assertFalse(_recalculate)
        self.assertListEqual(step_index, _diff_step_idx)

    def test_WithAngularSpectrumDiffractionCase2(self):
        num_steps = 10
        step_index = [0] * num_steps
        _diff_step_idx, _equidistant_steps, _recalculate = \
            adjust_equidistant_steps(AngularSpectrumDiffraction,
                                     False,
                                     num_steps,
                                     step_index)
        self.assertFalse(_equidistant_steps)
        self.assertFalse(_recalculate)
        self.assertListEqual(step_index, _diff_step_idx)
