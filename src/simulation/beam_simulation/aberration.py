# -*- coding: utf-8 -*-
"""
    aberration.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
from typing import Optional

import numpy

from simulation.controls import consts
from simulation.controls.main_control import MainControl
from system.material.aberration_phantom import AberrationPhantom


def aberration(control: MainControl,
               phantom: Optional[AberrationPhantom] = None) -> numpy.ndarray:
    """
    Creates a set of time delays based on the body wall model specified by heterogeneity.

    :param control: control
    :param phantom:
        Optional. The aberration phantom as an (num spheres x 4) array
        where the columns are sorted as [x y z R].
    :return:  A set of time delays
    """
    heterogeneous_medium = control.heterogeneous_medium
    if phantom is None:
        heterogeneous_medium = consts.ABERRATION_FROM_DELAY_SCREEN_BODY_WALL

    # pylint: disable=no-else-raise
    if heterogeneous_medium in (consts.ABERRATION_FROM_DELAY_SCREEN_BODY_WALL,
                                consts.ABERRATION_FROM_FILE):
        raise NotImplementedError
        # pylint: disable=no-else-raise
    elif heterogeneous_medium is consts.ABERRATION_PHANTOM:
        raise NotImplementedError
    else:
        delta = numpy.array(0.0)

    return delta
