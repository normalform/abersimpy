from backup.abersim_io import STRINGISNULL, FILEISNULL, VARISNULL
from scipy.io import loadmat


def load_propcontrol0(filename, prop):
    if filename == '' or filename == None:
        return STRINGISNULL

    try:
        data = loadmat(filename, squeeze_me=True)
        if 'Propcontrol' not in data:
            return VARISNULL
        mprop = data['Propcontrol']

        prop.simname = mprop['simname'].item()

        prop.ndims = mprop['ndims'].item()
        prop.nx = mprop['nx'].item()
        prop.ny = mprop['ny'].item()
        prop.nt = mprop['nt'].item()
        prop.PMLwidth = mprop['PMLwidth'].item()

        prop.diffrflag = mprop['diffrflag'].item()
        prop.nonlinflag = mprop['nonlinflag'].item()
        prop.lossflag = mprop['lossflag'].item()
        prop.abflag = mprop['abflag'].item()
        prop.annflag = mprop['annflag'].item()
        prop.equidistflag = mprop['equidistflag'].item()
        prop.historyflag = mprop['historyflag'].item()

        prop.stepsize = mprop['stepsize'].item()
        prop.nwindow = mprop['nwindow'].item()
        prop.shockstep = mprop['shockstep'].item()
        prop.endpoint = mprop['endpoint'].item()
        prop.currentpos = mprop['currentpos'].item()
        prop.storepos = mprop['storepos'].item()

        mmat = mprop['material'].item()
        prop.mat.i = mmat['i'].item()
        prop.mat.temp = mmat['temp'].item()
        prop.mat.eps = mmat['eps'].item()[0]
        prop.mat.c0 = mmat['c0'].item()
        prop.mat.rho = mmat['rho'].item()
        prop.mat.kappa = mmat['kappa'].item()
        prop.mat.betan = mmat['betan'].item()
        prop.mat.a = mmat['a'].item()
        prop.mat.b = mmat['b'].item()

        prop.d = mprop['d'].item()
        prop.offset = int(mprop['offset'].item()[0])
        prop.numscreens = mprop['numscreens'].item()
        prop.abamp = mprop['abamp'].item()[0]
        prop.ablength = mprop['ablength'].item()[0]
        prop.abfile = mprop['abfile'].item()

        prop.Fs = mprop['Fs'].item()
        prop.dx = mprop['dx'].item()
        prop.dy = mprop['dy'].item()
        prop.dt = mprop['dt'].item()
        prop.dz = mprop['dz'].item()
        prop.fc = mprop['fc'].item()
        prop.bandwidth = mprop['bandwidth'].item()
        prop.Np = mprop['Np'].item()
        prop.amplitude = mprop['amplitude'].item()
        prop.harmonic = mprop['harmonic'].item()
        prop.filter = mprop['filter'].item()

        prop.Dx = mprop['Dx'].item()
        prop.Dy = mprop['Dy'].item()
        prop.Fx = mprop['Fx'].item()
        prop.Fy = mprop['Fy'].item()
        prop.cchannel = mprop['cchannel'].item()
        prop.Nex = mprop['Nex'].item()
        prop.Ney = mprop['Ney'].item()
        prop.esizex = mprop['esizex'].item()
        prop.esizey = mprop['esizey'].item()

        return prop

    except FileNotFoundError:
        return FILEISNULL

