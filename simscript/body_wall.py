"""
body_wall.py
"""
import numpy

from controls.consts import AberrationFromDelayScreenBodyWall
from heterogeneous.aberration import aberration
from propagation.get_wave_numbers import get_wave_numbers


def body_wall(control,
              u_z,
              dir,
              equidistant_steps,
              wave_numbers=None,
              w=None,
              phantom=None):
    if wave_numbers is not None:
        _wave_numbers = get_wave_numbers(control, equidistant_steps)
    else:
        _wave_numbers = wave_numbers

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
