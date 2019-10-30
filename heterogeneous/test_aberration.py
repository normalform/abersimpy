# pylint: disable-all

import unittest
from unittest.mock import Mock
import numpy

from heterogeneous.aberration import aberration
from consts import NoAberrationAndHomogeneousMedium, \
    AberrationFromDelayScreenBodyWall, AberrationFromFile, AberrationPhantom


class TestAberration(unittest.TestCase):
    def test_WithPhantomButNoAberration(self):
        prop_control = Mock()
        prop_control.config.heterogeneous_medium = NoAberrationAndHomogeneousMedium
        phantom = Mock()
        delta = aberration(prop_control, phantom)

        self.assertEqual(numpy.array(0.0), delta)

    def test_WithPhantomAndAberrationFromDelayScreenBodyWall_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            prop_control = Mock()
            prop_control.config.heterogeneous_medium = AberrationFromDelayScreenBodyWall
            aberration(prop_control)

    def test_WithPhantomAndAberrationFromFile_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            prop_control = Mock()
            prop_control.config.heterogeneous_medium = AberrationFromFile
            aberration(prop_control)

    def test_WithPhantomAndAberrationPhantom_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            prop_control = Mock()
            prop_control.config.heterogeneous_medium = AberrationPhantom
            aberration(prop_control)

    def test_WithoutPhantom_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            prop_control = Mock()
            prop_control.config.heterogeneous_medium = NoAberrationAndHomogeneousMedium
            aberration(prop_control)
