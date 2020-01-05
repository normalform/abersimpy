"""
diffraction.py

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

from system.diffraction.interfaces import IDiffractionType


class NoDiffraction(IDiffractionType):
    """
    No diffraction
    """


class ExactDiffraction(IDiffractionType):
    """
    Exact diffraction using angular spectrum with wave number operator(Kz) as a variable
    """


class AngularSpectrumDiffraction(IDiffractionType):
    """
    Angular spectrum with wave number operator(Kz) as vectors (saves memory)
    """


class PseudoDifferential(IDiffractionType):
    """
    Pseudo differential model using matrix diagonalization for decoupling of equations
    """


class FiniteDifferenceTimeDifferenceReduced(IDiffractionType):
    """
    Using a finite difference time
    difference scheme in time and space with the parabolic
    approximation. The matrices used for differentiation
    are banded to improve computation time.
    """


class FiniteDifferenceTimeDifferenceFull(IDiffractionType):
    """
    Using a finite difference time
    difference scheme in time and space with the parabolic
    approximation. The matrices used for differentiation
    are full and requires more computation time.
    """
