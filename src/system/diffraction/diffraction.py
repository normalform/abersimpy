# -*- coding: utf-8 -*-
"""
    diffraction.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
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
