from backup.propio import load_propcontrol
from backup.propcontrol import init_propcontrol0
from backup.prepare_aberration import prepare_aberration
from backup.phantom import Phantom
from backup.abersim_math import ZSCALE, DTOL

import numpy as np


def plan_and_run_simulation(fnstr, logstr):
    #Load in Propcontrol
    prop = init_propcontrol0()
    load_propcontrol(fnstr, prop, logstr)

    # Assign flags
    diffrflag = prop.diffrflag
    nonlinflag = prop.nonlinflag
    lossflag = prop.lossflag
    abflag = prop.abflag
    annflag = prop.annflag
    equidistflag = prop.equidistflag
    historyflag = prop.historyflag

    nx = prop.nx
    ny = prop.ny
    nt = prop.nt

    ds = [0.0] * 4
    ds[0] = prop.dx
    ds[1] = prop.dy
    ds[2] = prop.dt
    ds[3] = prop.dz
    print("Initializing variables")

    # Preparing body wall
    stepsize = prop.stepsize
    abstepsize = 0.0
    delta = None
    if abflag == 0:
        abstepsize = stepsize
    elif abflag == 1:
        delta = [0.0] * nx * ny * prop.numscreens
        dscreen = prop.d / float(prop.numscreens)
        relstep = dscreen / stepsize
        if relstep < 1.0:
            abstepsize = dscreen
        else:
            abstepsize = dscreen / np.ceil(relstep)

    elif abflag == 2:
        print('Aberration type not yet available')
    else:
        print('Invalid abflag, setting abflag=0')
        abflag = 0
        prop.abflag = 0
        abstepsize = stepsize
    print('Initializing aberration')

    phantom = Phantom()
    prepare_aberration(delta, phantom, prop, logstr)

    # Adjusting stepsizes etc for nonlinear propagation
    nonstepsize = 0.0
    ddiff = 0.0
    if nonlinflag != 0 or diffrflag > 3:
        nsubs = np.ceil(abstepsize / prop.dz)
        nonstepsize = abstepsize / (ZSCALE * nsubs)
        c = prop.mat.c0
        ddiff = (c / 2.0) * (prop.dt / 2.0) * nonstepsize / ZSCALE
    else:
        nonstepsize = stepsize
        ddiff = stepsize
    print('Initializing nonlinear parameters')

    if equidistflag != 0 and np.abs(abstepsize - stepsize) > DTOL:
        equidistflag = 0
    print('Checked equidistflag')

    flags = [0] * 7
    flags[0] = diffrflag
    flags[1] = nonlinflag
    flags[2] = lossflag
    flags[3] = abflag
    flags[4] = annflag
    flags[5] = equidistflag
    flags[6] = historyflag

    globn = [0] * 3
    locn = [0] * 3
    globn[0] = nx
    globn[1] = ny
    globn[2] = nt
    locn[0] = nx
    locn[1] = ny
    locn[2] = nt

    # Allocating the right diffraction operator
    cuz = [np.complex()] * nx * ny * nt
    print('Allocated wave field')

    FFTP = 0
    if diffrflag == 0:
        print('Diffraction switched off')
    if diffrflag == 1 or diffrflag == 2:
        if prop.ndims == 2:
            cuz = np.array(cuz).reshape(nx, nt)
            cuz = np.fft.fft2(cuz)
        else:
            pass
    if diffrflag == 3:
        pass
    if diffrflag == 4 or diffrflag == 5:
        pass
    else:
        pass