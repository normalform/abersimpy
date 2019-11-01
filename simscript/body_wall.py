"""
body_wall.py
"""
import numpy

from controls.consts import AberrationFromDelayScreenBodyWall
from heterogeneous.aberration import aberration
from propagation.get_wavenumbers import get_wavenumbers


def body_wall(u_z,
              dir,
              control,
              equidistant_steps,
              Kz=None,
              w=None,
              phantom=None):
    global KZ
    if 'Kz' in globals():
        KZ = Kz
        del Kz

    if KZ is not None:
        KZ = get_wavenumbers(control, equidistant_steps)

    # Initiates profiles
    rmspro = None
    maxpro = None
    axplse = None
    zpos = None

    # Initiate variables
    num_dimensions = control.num_dimensions
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t
    heterogeneous_medium = control.heterogeneous_medium
    annular_transducer = control.annular_transducer
    resolution_x = control.signal.resolution_x
    resolution_y = control.signal.resolution_y

    step_size = control.simulation.step_size
    ns = control.material.num_screens
    thickness = control.material.thickness
    endpoint = numpy.min(control.simulation.endpoint, thickness)

    # adjust step sizes
    if heterogeneous_medium == AberrationFromDelayScreenBodyWall or phantom is None:
        if phantom is None:
            print('Changing heterogeneous_medium to AberrationFromDelayScreenBodyWall')
            heterogeneous_medium = AberrationFromDelayScreenBodyWall
            control.heterogeneous_medium = heterogeneous_medium

    dscreen = thickness / ns
    nsubsteps = numpy.ceil(dscreen / step_size)
    abstepsize = dscreen / nsubsteps
    control.simulation.step_size = abstepsize

    # Prepares body wall model
    delta = aberration(control)

    raise NotImplementedError
