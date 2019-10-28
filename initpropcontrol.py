from material.list_matrial import MUSCLE, ABPHANTOM
from material.check_material import check_material
from material.set_material import set_material
from log2roundoff import log2roundoff
from propcontrol import PropControl

import numpy


def initpropcontrol(fn = 'beamsim',
                    ndim = 2,
                    flags = [1, 1, 1, 0, 0, 0, 2],
                    harm = 2,
                    fi = 3,
                    bandwidth = 0.5,
                    np = 0,
                    amp = 0.5,
                    material = MUSCLE,
                    temp = 37.0,
                    endpoint = 0.1,
                    faz = 0.06,
                    fel = [],
                    nelaz = None,
                    dimelaz = None,
                    nelel = None,
                    dimelel = None):
    if np == 0:
        np = 4 * numpy.sqrt(numpy.log(2)) / (numpy.pi * bandwidth)

    diffrflag = flags[0]
    nonlinflag = flags[1]
    lossflag = flags[2]
    abflag = flags[3]
    annflag = flags[4]
    equidistflag = flags[5]
    historyflag = flags[6]

    if annflag != 0:
        if abflag != 0:
            diffrflag = 1
            ndim = 3
        elif diffrflag < 3:
            diffrflag = 3
            ndim = 2
        else:
            ndim = 2
    if harm > 1:
        nonlinflag = 1

    if nelaz is None:
        if annflag != 0:
            nelaz = 8
        else:
            nelaz = 64
    if dimelaz is None:
        if annflag != 0:
            dimelaz = 13e-4
        else:
            dimelaz = 3.5e-4
    if nelel is None:
        if annflag != 0:
            nelel = 8
        else:
            nelel = 1
    if dimelel is None:
        if annflag != 0:
            dimelel = 13e-4
        else:
            dimelel = 0.012

    # assign material struct
    ret = check_material(material)
    if not ret:
        print('Material is not properly specified.')
        print('MUSCLE at {} degrees is used'.format(temp))
        material = set_material(MUSCLE, temp)
    else:
        material = set_material(material, temp)

    # adjust frequency dependent variables
    fs = numpy.array([0.1, 0.5, 1.5, 3.0, 6.0, 12.0]) * 1e6
    ss = numpy.array([10, 5, 2.5, 1.25, 0.5, 0.25]) * 1e-3
    filter = numpy.array([1.0, 1.6, 2.0, 2.0, 2.0, 2.2, 2.2, 2.2, 2.2, 2.2]) / numpy.arange(1, 11) * 0.5
    fi = fi * 1e6
    ft = fi / harm
    c = material.c0
    lambdai = c / fi
    if ndim == 2:
        scale = 2
    else:
        scale = 1

    dx = lambdai / (2 * scale)
    ndxprel = numpy.ceil(dimelaz / dx)
    if numpy.mod(ndxprel, 2) == 0 and annflag != 0:
        ndxprel = ndxprel + 1
    dx = dimelaz / ndxprel
    if annflag != 0:
        daz = (2 * nelaz - 1) * dimelaz
    else:
        daz = nelaz * dimelaz

    dy = lambdai / (2 * scale)
    ndyprel = numpy.ceil(dimelel / dy)
    if numpy.mod(ndyprel, 2) == 0 and annflag != 0:
        ndyprel = ndyprel + 1
    dy = dimelel / ndyprel
    if annflag != 0:
        _del = (2 * nelel - 1) * dimelel
    else:
        _del = nelel * dimelel

    idx = numpy.where(abs(fi - fs) == min(abs(fi-fs)))[0][-1]
    stepsize = ss[idx]

    # calculate domain specific variables
    if not fel:
        fel = faz
    elif fel != (faz and annflag):
        fel = faz

    if ndim == 1:
        nlambdapad = 0
        nperiods = 12
    elif ndim == 2:
        nlambdapad = 35
        nperiods = 12
    elif ndim == 3:
        nlambdapad = 25
        nperiods = 8

    omegax= daz + 2 * nlambdapad * lambdai
    nptx = omegax / dx
    nx = log2roundoff(nptx)
    if annflag and diffrflag >= 3:
        nx = nx / 2
    if ndim == 3 and diffrflag < 3:
        omegay = _del + 2 * nlambdapad * lambdai
        npty = omegay / dy
        ny = log2roundoff(npty)
    else:
        ny = 1
    if ndim == 1:
        nx = 1
        ny = 1

    Fs = numpy.maximum(40e6, 10.0 * ft)
    dt = 1.0 / Fs
    nptt = nperiods * (np / ft) / dt
    nt = log2roundoff(nptt)

    # simulation name
    propcontrol = PropControl()
    propcontrol.simname = fn

    # domain and grid specifications
    propcontrol.ndims = ndim
    propcontrol.nx = nx
    propcontrol.ny = ny
    propcontrol.nt = nt
    propcontrol.PMLwidth = 0

    # flags used for propagation
    propcontrol.diffrflag = diffrflag
    propcontrol.nonlinflag = nonlinflag
    propcontrol.lossflag = lossflag
    propcontrol.abflag = abflag
    propcontrol.annflag = annflag
    propcontrol.equidistflag = equidistflag
    propcontrol.historyflag = historyflag

    # simulation parameters
    propcontrol.stepsize = stepsize
    propcontrol.nwindow = 2
    propcontrol.shockstep = 0.5
    propcontrol.endpoint = endpoint
    propcontrol.currentpos = 0.0
    propcontrol.storepos = [fel, faz]

    # material parameters
    propcontrol.material = material
    if material.i == ABPHANTOM:
        propcontrol.d = 0.035
    else:
        propcontrol.d = 0.02
    propcontrol.offset = [0, 0]
    ns = 8
    propcontrol.numscreens = ns
    propcontrol.abamp = 0.09 * numpy.ones((ns, 1)) * 1e-3
    propcontrol.ablength = numpy.ones((ns, 1)) * 1e-3 * numpy.array([4, 100])
    propcontrol.abseed = numpy.arange(1, ns+1)
    if abflag == 2:
        propcontrol.abfile = 'randseq.mat'
    elif abflag == 3:
        propcontrol.abfile = 'phantoml.mat'
    else:
        propcontrol.abfile = ''

    # signal parameters
    propcontrol.Fs = Fs
    propcontrol.dx = dx
    propcontrol.dy = dy
    propcontrol.dz = c / (2.0 * numpy.pi * fi)
    propcontrol.dt = dt
    propcontrol.fc = ft
    propcontrol.bandwidth = bandwidth * ft
    propcontrol.np = np
    propcontrol.amplitude = amp
    propcontrol.harmonic = harm
    propcontrol.filter = filter[0:harm]

    # transducer parameters
    propcontrol.Dx = daz
    propcontrol.Dy = _del
    propcontrol.Fx = faz
    propcontrol.Fy = fel
    if diffrflag >= 3 and annflag != 0:
        propcontrol.cchannel = numpy.array([1, 1])
    else:
        propcontrol.cchannel = numpy.floor([propcontrol.nx / 2, propcontrol.ny / 2]) + 1
    propcontrol.cchannel.astype(int)
    propcontrol.Nex = nelaz
    propcontrol.Ney = nelel
    propcontrol.esizex = dimelaz
    propcontrol.esizey = dimelel

    if abflag != 0:
        propcontrol.storepos = numpy.array([propcontrol.storepos, propcontrol.d])
    propcontrol.storepos = numpy.unique(propcontrol.storepos)

    return propcontrol
