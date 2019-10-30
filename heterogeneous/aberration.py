"""
aberration.py
"""
import numpy

from consts import AberrationFromDelayScreenBodyWall, AberrationFromFile, AberrationPhantom
from init_prop_control import PropControl


def aberration(prop_control: PropControl,
               phantom=None) -> numpy.ndarray:
    """
    Creates a set of time delays based on the body wall model specified by heterogeneity.

    :param prop_control: prop control
    :param phantom:
        Optional. The aberration phantom as an (num spheres x 4) array
        where the columns are sorted as [x y z R].
    :return:  A set of time delays
    """
    heterogeneous_medium = prop_control.config.heterogeneous_medium
    if phantom is None:
        heterogeneous_medium = AberrationFromDelayScreenBodyWall

    # pylint: disable=no-else-raise
    if heterogeneous_medium in (AberrationFromDelayScreenBodyWall, AberrationFromFile):
        raise NotImplementedError
        # pylint: disable=no-else-raise
    elif heterogeneous_medium == AberrationPhantom:
        raise NotImplementedError
    else:
        delta = numpy.array(0.0)

    return delta
