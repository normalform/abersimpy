# pylint: disable-all

import unittest
from unittest.mock import Mock

import numpy

from controls import consts
from heterogeneous.aberration import aberration


class TestAberration(unittest.TestCase):
    def test_WithPhantomButNoAberration(self):
        main_control = Mock()
        main_control.config.heterogeneous_medium = consts.NoAberrationAndHomogeneousMedium
        phantom = Mock()
        delta = aberration(main_control, phantom)

        self.assertEqual(numpy.array(0.0), delta)

    def test_WithPhantomAndAberrationFromDelayScreenBodyWall_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            main_control = Mock()
            main_control.config.heterogeneous_medium = consts.AberrationFromDelayScreenBodyWall
            aberration(main_control)

    def test_WithPhantomAndAberrationFromFile_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            main_control = Mock()
            main_control.config.heterogeneous_medium = consts.AberrationFromFile
            aberration(main_control)

    def test_WithPhantomAndAberrationPhantom_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            main_control = Mock()
            main_control.config.heterogeneous_medium = consts.AberrationPhantom
            aberration(main_control)

    def test_WithoutPhantom_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            main_control = Mock()
            main_control.config.heterogeneous_medium = consts.NoAberrationAndHomogeneousMedium
            aberration(main_control)
