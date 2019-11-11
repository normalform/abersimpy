# pylint: disable-all

import unittest
from unittest.mock import Mock

import numpy

from controls import consts
from simulation.aberration import aberration


class TestAberration(unittest.TestCase):
    def test_WithPhantomButNoAberration(self):
        control = Mock()
        control.heterogeneous_medium = consts.NoAberrationAndHomogeneousMedium
        phantom = Mock()
        delta = aberration(control, phantom)

        self.assertEqual(numpy.array(0.0), delta)

    def test_WithPhantomAndAberrationFromDelayScreenBodyWall_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            control = Mock()
            control.heterogeneous_medium = consts.AberrationFromDelayScreenBodyWall
            aberration(control)

    def test_WithPhantomAndAberrationFromFile_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            control = Mock()
            control.heterogeneous_medium = consts.AberrationFromFile
            aberration(control)

    def test_WithPhantomAndAberrationPhantom_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            control = Mock()
            control.heterogeneous_medium = consts.AberrationPhantom
            aberration(control)

    def test_WithoutPhantom_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            control = Mock()
            control.heterogeneous_medium = consts.NoAberrationAndHomogeneousMedium
            aberration(control)
