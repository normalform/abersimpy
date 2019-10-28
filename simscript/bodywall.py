from initpropcontrol import initpropcontrol
from propagation.get_wavenumbers import get_wavenumbers
from heterogeneous.aberrator import aberrator

import numpy


def bodywall(u_z,
             dir,
             propcontrol = None,
             Kz = None,
             w = None,
             phantom = None):
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
    ndim = propcontrol.ndims
    nx = propcontrol.nx
    ny = propcontrol.ny
    nt = propcontrol.nt
    abflag = propcontrol.abflag
    annflag = propcontrol.annflag
    dx = propcontrol.dx
    dy = propcontrol.dy

    stepsize = propcontrol.stepsize
    ns = propcontrol.numscreens
    d = propcontrol.d
    endpoint = numpy.min(propcontrol.endpoint, d)

    # adjust stepsizes
    if abflag == 1 or phantom is None:
        if phantom is None:
            print('Changing abflag to 1')
            abflag = 1
            propcontrol.abflag = abflag

    dscreen = d / ns
    nsubsteps = numpy.ceil(dscreen / stepsize)
    abstepsize = dscreen / nsubsteps
    propcontrol.stepsize = abstepsize

    # Prepares bodywall model
    delta = aberrator(propcontrol)

    # TODO
    raise NotImplementedError