"""
aberration.py
"""
import numpy

from controls import consts
from controls.main_control import MainControl


def aberration(main_control: MainControl,
               phantom=None) -> numpy.ndarray:
    """
    Creates a set of time delays based on the body wall model specified by heterogeneity.

    :param main_control: prop control
    :param phantom:
        Optional. The aberration phantom as an (num spheres x 4) array
        where the columns are sorted as [x y z R].
    :return:  A set of time delays
    """
    heterogeneous_medium = main_control.config.heterogeneous_medium
    if phantom is None:
        heterogeneous_medium = consts.AberrationFromDelayScreenBodyWall

    # pylint: disable=no-else-raise
    if heterogeneous_medium in (consts.AberrationFromDelayScreenBodyWall, consts.AberrationFromFile):
        raise NotImplementedError
        # pylint: disable=no-else-raise
    elif heterogeneous_medium == consts.AberrationPhantom:
        raise NotImplementedError
    else:
        delta = numpy.array(0.0)

    return delta
