"""
body_wall.py
"""
import numpy

from consts import AberrationFromDelayScreenBodyWall
from heterogeneous.aberration import aberration
from prop_control import PropControl
from propagation.get_wavenumbers import get_wavenumbers


def body_wall(u_z,
              dir,
              prop_control=None,
              Kz=None,
              w=None,
              phantom=None):
    if prop_control is None:
        prop_control = PropControl.init_prop_control()

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
    nx = prop_control.nx
    ny = prop_control.ny
    nt = prop_control.nt
    heterogeneous_medium = prop_control.config.heterogeneous_medium
    annular_transducer = prop_control.annular_transducer
    dx = prop_control.dx
    dy = prop_control.dy

    stepsize = prop_control.stepsize
    ns = prop_control.numscreens
    d = prop_control.d
    endpoint = numpy.min(prop_control.endpoint, d)

    # adjust stepsizes
    if heterogeneous_medium == AberrationFromDelayScreenBodyWall or phantom is None:
        if phantom is None:
            print('Changing heterogeneous_medium to AberrationFromDelayScreenBodyWall')
            heterogeneous_medium = AberrationFromDelayScreenBodyWall
            prop_control.config.heterogeneous_medium = heterogeneous_medium

    dscreen = d / ns
    nsubsteps = numpy.ceil(dscreen / stepsize)
    abstepsize = dscreen / nsubsteps
    prop_control.stepsize = abstepsize

    # Prepares body wall model
    delta = aberration(prop_control)

    # TODO
    raise NotImplementedError