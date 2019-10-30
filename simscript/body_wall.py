"""
body_wall.py
"""
import numpy

from initpropcontrol import initpropcontrol
from propagation.get_wavenumbers import get_wavenumbers
from heterogeneous.aberration import aberration
from consts import AberrationFromDelayScreenBodyWall


def body_wall(u_z,
              dir,
              propcontrol=None,
              Kz=None,
              w=None,
              phantom=None):
    if propcontrol is None:
        propcontrol = initpropcontrol()

    global KZ
    if 'Kz' in globals():
        KZ = Kz
        del Kz

    if KZ is not None:
        KZ  = get_wavenumbers(propcontrol)

    # Initiates profiles
    rmspro = None
    maxpro = None
    axplse = None
    zpos = None

    # Initiate variables
    num_dimensions = propcontrol.num_dimensions
    nx = propcontrol.nx
    ny = propcontrol.ny
    nt = propcontrol.nt
    heterogeneous_medium = propcontrol.config.heterogeneous_medium
    annular_transducer = propcontrol.annular_transducer
    dx = propcontrol.dx
    dy = propcontrol.dy

    stepsize = propcontrol.stepsize
    ns = propcontrol.numscreens
    d = propcontrol.d
    endpoint = numpy.min(propcontrol.endpoint, d)

    # adjust stepsizes
    if heterogeneous_medium == AberrationFromDelayScreenBodyWall or phantom is None:
        if phantom is None:
            print('Changing heterogeneous_medium to AberrationFromDelayScreenBodyWall')
            heterogeneous_medium = AberrationFromDelayScreenBodyWall
            propcontrol.config.heterogeneous_medium = heterogeneous_medium

    dscreen = d / ns
    nsubsteps = numpy.ceil(dscreen / stepsize)
    abstepsize = dscreen / nsubsteps
    propcontrol.stepsize = abstepsize

    # Prepares body wall model
    delta = aberration(propcontrol)

    # TODO
    raise NotImplementedError