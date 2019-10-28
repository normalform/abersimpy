from abersim_size import NS
from backup.material import Material


class PropControl(object):
    simname = None

    ndims = 0

    nx = 0
    ny = 0
    nt = 0

    plm_width = 0

    diffrflag = 0
    nonlinflag = 0
    lossflag = 0
    abflag = 0
    annflag = 0
    equidistflag = 0
    historyflag = 0

    stepsize = 0.0
    nwindow = 0.0
    shockstep = 0.0
    endpoint = 0.0
    currentpos = 0.0
    nstorepos = 0
    storepos = 0.0

    mat = None

    d = 0.0
    offset = [0.0] * 2
    numscreens = 1
    abamp = [0.0] * NS
    ablength = [0.0] * NS * 2
    abseed = [0] * NS

    abfile = ''

    Fs = 0
    dx = 0.0
    dy = 0.0
    dz = 0.0
    dt = 0.0
    fc = 0.0
    bandwidth = 0.0
    Np = 0.0
    amplitude = 0.0
    harmonic = 0
    filter = [0.0] * 10

    Dx = 0.0
    Dy = 0.0
    Fx = 0.0
    Fy = 0.0
    cc = [0] * 2
    Nex = 0
    Ney = 0
    esizex = 0.0
    esizey = 0.0


def init_propcontrol0():
    prop = PropControl()

    prop.simname = None

    prop.ndims = 0
    prop.nx = 0
    prop.ny = 0
    prop.nt = 0
    prop.PMLwidth = 0

    prop.diffrflag = 0
    prop.nonlinflag = 0
    prop.lossflag = 0
    prop.abflag = 0
    prop.annflag = 0
    prop.equidistflag = 0
    prop.historyflag = 0

    prop.stepsize = 0.0
    prop.nwindow = 0.0
    prop.shockstep = 0.0
    prop.endpoint = 0.0
    prop.currentpos = 0.0
    prop.nstorepos = 0
    prop.storepos = None

    prop.mat = Material()
    prop.mat.i = 0
    prop.mat.reg = 0
    prop.mat.temp = 0.0
    prop.mat.eps[0] = 0.0
    prop.mat.eps[1] = 0.0
    prop.mat.eps[2] = 0.0
    prop.mat.c0 = 0.0
    prop.mat.rho = 0.0
    prop.mat.kappa = 0.0
    prop.mat.betan = 0.0
    prop.mat.a = 0.0
    prop.mat.b = 0.0

    prop.d = 0.0
    prop.offset[0] = 0.0
    prop.offset[1] = 0.0
    prop.numscreens = 0
    for i in range(NS):
        prop.abamp[i] = 0.0
    for i in range(NS * 2):
        prop.ablength[i] = 0.0
    for i in range(NS):
        prop.abseed[i] = 0
    prop.abfile = None

    prop.Fs = 0.0
    prop.dx = 0.0
    prop.dy = 0.0
    prop.dt = 0.0
    prop.dz = 0.0
    prop.fc = 0.0
    prop.bandwidth = 0.0
    prop.Np = 0.0
    prop.amplitude = 0.0
    prop.harmonic = 0
    for i in range(10):
        prop.filter[i] = 0.0

    prop.Dx = 0.0
    prop.Dy = 0.0
    prop.Fx = 0.0
    prop.Fy = 0.0
    prop.cc[0] = 0
    prop.cc[1] = 0
    prop.Nex = 0
    prop.Ney = 0
    prop.esizex = 0.0
    prop.esizey = 0.0

    return prop