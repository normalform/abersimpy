import numpy
from scipy.signal import hilbert

from controls.consts import ScaleForTemporalVariable, ScaleForSpatialVariablesZ
from controls.prop_control import NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from filter.get_freqs import get_freqs


def get_wavenumbers(prop_control,
                    noret=0):
    num_points_x = prop_control.num_points_x
    num_points_y = prop_control.num_points_y
    num_points_t = prop_control.num_points_t
    mat = prop_control.material
    c = mat.sound_speed
    resolution_t = prop_control.resolution_t
    ft = 1 / resolution_t
    df = ft / num_points_t
    kt = 2.0 * numpy.pi / c * numpy.arange(-ft / 2, ft / 2, df)

    if prop_control.config.diffraction_type == NoDiffraction or \
            prop_control.config.diffraction_type == ExactDiffraction or \
            prop_control.config.diffraction_type == AngularSpectrumDiffraction:
        fx = 1 / prop_control.resolution_x
        fy = 1 / prop_control.resolution_y
        dkx = fx / num_points_x
        dky = fy / num_points_y
        if num_points_x == 1:
            kx = 0
        else:
            kx = 2.0 * numpy.pi * numpy.arange(-fx / 2, fx / 2, dkx)
        if num_points_y == 1:
            ky = 0
        else:
            ky = 2.0 * numpy.pi * numpy.arange(-fy / 2, fy / 2, dky)
    elif prop_control.config.diffraction_type == PseudoDifferential:
        raise NotImplementedError
    elif prop_control.config.diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
            prop_control.config.diffraction_type == FiniteDifferenceTimeDifferenceFull:
        raise NotImplementedError

    # calculate attenuation if propagation is linear
    loss = numpy.zeros((kt.size))
    if prop_control.config.attenuation and prop_control.config.non_linearity is False:
        w = get_freqs(num_points_t, prop_control.resolution_t / (2.0 * numpy.pi * ScaleForTemporalVariable))
        epsa = mat.eps_a
        epsb = mat.eps_b
        loss = epsa * numpy.conj(hilbert(numpy.abs(w) ** epsb)) / ScaleForSpatialVariablesZ

    # assembly of wave-number operator
    if prop_control.config.diffraction_type == AngularSpectrumDiffraction:
        # assign to vectors
        Kz = numpy.zeros((numpy.max((num_points_x, num_points_y, num_points_t)), 4))
        Kz[:num_points_x, 0] = numpy.fft.ifftshift(kx)
        Kz[:num_points_y, 1] = numpy.fft.ifftshift(ky)
        Kz[:num_points_t, 2] = numpy.fft.ifftshift(kt)
        Kz[:num_points_t, 3] = loss
        return Kz
    else:
        # building full complex wave number operator
        if prop_control.num_dimensions == 3:
            Ky2, Kx2 = numpy.meshgrid(numpy.fft.ifftshift(ky ** 2), numpy.fft.ifftshift(kx ** 2), indexing='ij')
            Kxy2 = Kx2 + Ky2
            Kxy2 = Kxy2.reshape((num_points_x * num_points_y, 1))
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
        for ii in range(0, num_points_t):
            Kz[ii, ...] = Kz[ii, ...] - 1j * loss[ii]

    # convert wave number operator to propagation operator
    if prop_control.config.equidistant_steps:
        step_size = prop_control.step_size
        if prop_control.config.non_linearity:
            nsubsteps = int(numpy.ceil(step_size / prop_control.resolution_z))
            step_size = step_size / nsubsteps
        Kz = numpy.exp(-1j * Kz * step_size)

    if prop_control.config.diffraction_type == PseudoDifferential:
        raise NotImplementedError

    return Kz
