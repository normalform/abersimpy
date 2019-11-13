"""
aberration.py
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
    _heterogeneous_medium = control.heterogeneous_medium
    if phantom is None:
        _heterogeneous_medium = consts.AberrationFromDelayScreenBodyWall

    # pylint: disable=no-else-raise
    if _heterogeneous_medium in (consts.AberrationFromDelayScreenBodyWall,
                                 consts.AberrationFromFile):
        raise NotImplementedError
        # pylint: disable=no-else-raise
    elif _heterogeneous_medium is consts.AberrationPhantom:
        raise NotImplementedError
    else:
        delta = numpy.array(0.0)

    return delta
