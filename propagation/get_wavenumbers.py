import numpy
from scipy.signal import hilbert

from consts import ScaleForTemporalVariable, ScaleForSpatialVariablesZ
from filter.get_freqs import get_freqs
from prop_control import PropControl, NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull


def get_wavenumbers(prop_control=None,
                    noret = 0):
    if prop_control is None:
        prop_control = PropControl.init_prop_control()

    nx = prop_control.nx
    ny = prop_control.ny
    nt = prop_control.nt
    mat = prop_control.material
    c = mat.sound_speed
    dt = prop_control.dt
    ft = 1 / dt
    df = ft / nt
    kt = 2.0 * numpy.pi / c * numpy.arange(-ft/2, ft/2, df)

    if prop_control.config.diffraction_type == NoDiffraction or \
            prop_control.config.diffraction_type == ExactDiffraction or \
            prop_control.config.diffraction_type == AngularSpectrumDiffraction:
        fx = 1 / prop_control.dx
        fy = 1 / prop_control.dy
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
    elif prop_control.config.diffraction_type == PseudoDifferential:
        raise NotImplementedError
    elif prop_control.config.diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
            prop_control.config.diffraction_type == FiniteDifferenceTimeDifferenceFull:
        raise NotImplementedError

    # calculate attenuation if propagation is linear
    loss = numpy.zeros((kt.size))
    if prop_control.config.attenuation and prop_control.config.non_linearity is False:
        w = get_freqs(nt, prop_control.dt / (2.0 * numpy.pi * ScaleForTemporalVariable))
        epsa = mat.eps_a
        epsb = mat.eps_b
        loss = epsa * numpy.conj(hilbert(numpy.abs(w) ** epsb)) / ScaleForSpatialVariablesZ

    # assembly of wave-number operator
    if prop_control.config.diffraction_type == AngularSpectrumDiffraction:
        # assign to vectors
        Kz = numpy.zeros((numpy.max((nx, ny, nt)), 4))
        Kz[:nx, 0] = numpy.fft.ifftshift(kx)
        Kz[:ny, 1] = numpy.fft.ifftshift(ky)
        Kz[:nt, 2] = numpy.fft.ifftshift(kt)
        Kz[:nt, 3] = loss
        return Kz
    else:
        # building full complex wave number operator
        if prop_control.num_dimensions == 3:
            Ky2, Kx2 = numpy.meshgrid(numpy.fft.ifftshift(ky ** 2), numpy.fft.ifftshift(kx ** 2), indexing='ij')
            Kxy2 = Kx2 + Ky2
            Kxy2 = Kxy2.reshape((nx*ny, 1))
        elif prop_control.num_dimensions == 2:
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
    if prop_control.config.attenuation:
        for ii in range(0, nt):
            Kz[ii, ...] = Kz[ii, ...] - 1j*loss[ii]

    # convert wave number operator to propagation operator
    if prop_control.config.equidistant_steps:
        stepsize = prop_control.stepsize
        if prop_control.config.non_linearity:
            nsubsteps = int(numpy.ceil(stepsize / prop_control.dz))
            stepsize = stepsize / nsubsteps
        Kz = numpy.exp(-1j * Kz * stepsize)

    if prop_control.config.diffraction_type == PseudoDifferential:
        raise  NotImplementedError

    return Kz