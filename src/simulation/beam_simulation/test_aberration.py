"""
test_aberration.py

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
from unittest.mock import Mock

import numpy

from simulation.beam_simulation.aberration import aberration
from simulation.controls import consts


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
