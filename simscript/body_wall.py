"""
body_wall.py
"""
import numpy

from controls.consts import AberrationFromDelayScreenBodyWall
from heterogeneous.aberration import aberration
from propagation.get_wavenumbers import get_wavenumbers


def body_wall(u_z,
              dir,
              main_control,
              Kz=None,
              w=None,
              phantom=None):
    global KZ
    if 'Kz' in globals():
        KZ = Kz
        del Kz

    if KZ is not None:
        KZ = get_wavenumbers(main_control)

    # Initiates profiles
    rmspro = None
    maxpro = None
    axplse = None
    zpos = None

    # Initiate variables
    num_dimensions = main_control.num_dimensions
    num_points_x = main_control.num_points_x
    num_points_y = main_control.num_points_y
    num_points_t = main_control.num_points_t
    heterogeneous_medium = main_control.config.heterogeneous_medium
    annular_transducer = main_control.annular_transducer
    resolution_x = main_control.resolution_x
    resolution_y = main_control.resolution_y

    step_size = main_control.step_size
    ns = main_control.num_screens
    thickness = main_control.thickness
    endpoint = numpy.min(main_control.endpoint, thickness)

    # adjust step sizes
    if heterogeneous_medium == AberrationFromDelayScreenBodyWall or phantom is None:
        if phantom is None:
            print('Changing heterogeneous_medium to AberrationFromDelayScreenBodyWall')
            heterogeneous_medium = AberrationFromDelayScreenBodyWall
            main_control.config.heterogeneous_medium = heterogeneous_medium

    dscreen = thickness / ns
    nsubsteps = numpy.ceil(dscreen / step_size)
    abstepsize = dscreen / nsubsteps
    main_control.step_size = abstepsize

    # Prepares body wall model
    delta = aberration(main_control)

    # TODO
    raise NotImplementedError
