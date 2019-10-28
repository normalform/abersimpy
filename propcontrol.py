from abersim_size import NS
from material.material import Material

import json
import numpy


class PropControl(object):
    simname = None

    ndims = 0

    nx = 0
    ny = 0
    nt = 0

    PMLwidth = 0

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

    material = None

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
    cchannel = [0] * 2
    Nex = 0
    Ney = 0
    esizex = 0.0
    esizey = 0.0

class PropControlDataDecoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, PropControl):
            return object.__dict__
        if isinstance(object, numpy.ndarray):
            data = object.tolist()
            return data
        if isinstance(object, Material):
            return object.__dict__
        else:
            return json.JSONEncoder.default(self, object)
