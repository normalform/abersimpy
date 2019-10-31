"""
body_wall.py
"""
import numpy

from consts import AberrationFromDelayScreenBodyWall
from heterogeneous.aberration import aberration
from propagation.get_wavenumbers import get_wavenumbers


def body_wall(u_z,
              dir,
              prop_control,
              Kz=None,
              w=None,
              phantom=None):
    global KZ
    if 'Kz' in globals():
        KZ = Kz
        del Kz

    if KZ is not None:
        KZ = get_wavenumbers(prop_control)

    # Initiates profiles
    rmspro = None
    maxpro = None
    axplse = None
    zpos = None

    # Initiate variables
    num_dimensions = prop_control.num_dimensions
    num_points_x = prop_control.num_points_x
    num_points_y = prop_control.num_points_y
    num_points_t = prop_control.num_points_t
    heterogeneous_medium = prop_control.config.heterogeneous_medium
    annular_transducer = prop_control.annular_transducer
    resolution_x = prop_control.resolution_x
    resolution_y = prop_control.resolution_y

    step_size = prop_control.step_size
    ns = prop_control.num_screens
    thickness = prop_control.thickness
    endpoint = numpy.min(prop_control.endpoint, thickness)

    # adjust step sizes
    if heterogeneous_medium == AberrationFromDelayScreenBodyWall or phantom is None:
        if phantom is None:
            print('Changing heterogeneous_medium to AberrationFromDelayScreenBodyWall')
            heterogeneous_medium = AberrationFromDelayScreenBodyWall
            prop_control.config.heterogeneous_medium = heterogeneous_medium

    dscreen = thickness / ns
    nsubsteps = numpy.ceil(dscreen / step_size)
    abstepsize = dscreen / nsubsteps
    prop_control.step_size = abstepsize

    # Prepares body wall model
    delta = aberration(prop_control)

    # TODO
    raise NotImplementedError
