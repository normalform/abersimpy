# -*- coding: utf-8 -*-
"""
    test_aberration.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
# pylint: disable-all

import unittest
from unittest.mock import Mock

import numpy

from simulation.beam_simulation.aberration import aberration
from simulation.controls import consts
from system.material.aberration_phantom import AberrationPhantom


class TestAberration(unittest.TestCase):
    def test_WithPhantomButNoAberration(self):
        control = Mock()
        control.heterogeneous_medium = consts.NO_ABERRATION_AND_HOMOGENEOUS_MEDIUM
        phantom = Mock()
        delta = aberration(control, phantom)

        self.assertEqual(numpy.array(0.0), delta)

    def test_WithPhantomAndAberrationFromDelayScreenBodyWall_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            control = Mock()
            control.heterogeneous_medium = consts.ABERRATION_FROM_DELAY_SCREEN_BODY_WALL
            aberration(control)

    def test_WithPhantomAndAberrationFromFile_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            control = Mock()
            control.heterogeneous_medium = consts.ABERRATION_FROM_FILE
            aberration(control)

    def test_WithPhantomAndAberrationPhantom_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            control = Mock()
            control.heterogeneous_medium = consts.ABERRATION_PHANTOM
            aberration(control)

    def test_WithPhantomAndAberrationPhantomCase2_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            control = Mock()
            control.heterogeneous_medium = consts.ABERRATION_PHANTOM
            aberration(control, AberrationPhantom(37.0))

    def test_WithoutPhantom_Exception_NotSupported(self):
        with self.assertRaises(NotImplementedError):
            control = Mock()
            control.heterogeneous_medium = consts.NO_ABERRATION_AND_HOMOGENEOUS_MEDIUM
            aberration(control)
