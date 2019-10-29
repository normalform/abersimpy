from initpropcontrol import initpropcontrol
from filter.get_freqs import get_freqs
from consts import TSCALE, ZSCALE
from propcontrol import NoDiffraction,  ExactDiffraction, AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull

import numpy
from scipy.signal import hilbert


def get_wavenumbers(propcontrol = None,
                    noret = 0):
    if propcontrol is None:
        propcontrol = initpropcontrol()

    nx = propcontrol.nx
    ny = propcontrol.ny
    nt = propcontrol.nt
    mat = propcontrol.material
    c = mat.c0
    dt = propcontrol.dt
    ft = 1 / dt
    df = ft / nt
    kt = 2.0 * numpy.pi / c * numpy.arange(-ft/2, ft/2, df)

    if propcontrol.config.diffraction_type == NoDiffraction or \
            propcontrol.config.diffraction_type == ExactDiffraction or \
            propcontrol.config.diffraction_type == AngularSpectrumDiffraction:
        fx = 1 / propcontrol.dx
        fy = 1 / propcontrol.dy
        dkx = fx / nx
        dky = fy / ny
        if nx == 1:
            kx = 0
        else:
            kx = 2.0 * numpy.pi * numpy.arange(-fx/2, fx/2, dkx)
        if ny == 1:
            ky = 0
        else:
            ky = 2.0 * numpy.pi * numpy.arange(-fy/2, fy/2, dky)
    elif propcontrol.config.diffraction_type == PseudoDifferential:
        raise NotImplementedError
    elif propcontrol.config.diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
            propcontrol.config.diffraction_type == FiniteDifferenceTimeDifferenceFull:
        raise NotImplementedError

    # calculate attenuation if propagation is linear
    loss = numpy.zeros((kt.size))
    if propcontrol.config.attenuation and propcontrol.config.non_linearity is False:
        w = get_freqs(nt, propcontrol.dt / (2.0 * numpy.pi * TSCALE))
        epsa = mat.eps[1]
        epsb = mat.eps[2]
        loss = epsa * numpy.conj(hilbert(numpy.abs(w) ** epsb)) / ZSCALE

    # assembly of wave-number operator
    if propcontrol.config.diffraction_type == AngularSpectrumDiffraction:
        # assign to vectors
        Kz = numpy.zeros((numpy.max((nx, ny, nt)), 4))
        Kz[:nx, 0] = numpy.fft.ifftshift(kx)
        Kz[:ny, 1] = numpy.fft.ifftshift(ky)
        Kz[:nt, 2] = numpy.fft.ifftshift(kt)
        Kz[:nt, 3] = loss
        return Kz
    else:
        # building full complex wave number operator
        if propcontrol.num_dimensions == 3:
            Ky2, Kx2 = numpy.meshgrid(numpy.fft.ifftshift(ky ** 2), numpy.fft.ifftshift(kx ** 2), indexing='ij')
            Kxy2 = Kx2 + Ky2
            Kxy2 = Kxy2.reshape((nx*ny, 1))
        elif propcontrol.num_dimensions == 2:
            Kxy2 = numpy.fft.ifftshift(kx ** 2)
        else:
            Kxy2 = 0
    Kt, KXY = numpy.meshgrid(numpy.fft.ifftshift(kt), Kxy2, indexing='ij')
    Kz = numpy.sqrt((Kt ** 2 - KXY).astype(complex))
    Kz = numpy.sign(Kt) * Kz.real - 1j * Kz.imag

    # introduces retarded time
    if noret == 0:
        Kz = Kz - Kt

    # introduces loss in wave number operator
    if propcontrol.config.attenuation:
        for ii in range(0, nt):
            Kz[ii, ...] = Kz[ii, ...] - 1j*loss[ii]

    # convert wave number operator to propagation operator
    if propcontrol.config.equidistant_steps:
        stepsize = propcontrol.stepsize
        if propcontrol.config.non_linearity:
            nsubsteps = int(numpy.ceil(stepsize / propcontrol.dz))
            stepsize = stepsize / nsubsteps
        Kz = numpy.exp(-1j * Kz * stepsize)

    if propcontrol.config.diffraction_type == PseudoDifferential:
        raise  NotImplementedError

    return Kz