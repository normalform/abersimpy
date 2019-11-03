import numpy
from scipy.signal import hilbert

from controls.consts import ScaleForTemporalVariable, ScaleForSpatialVariablesZ
from controls.main_control import MainControl
from diffraction.diffraction import NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction, \
    PseudoDifferential, FiniteDifferenceTimeDifferenceFull, FiniteDifferenceTimeDifferenceReduced
from filter.get_frequencies import get_frequencies


def get_wave_numbers(control: MainControl,
                     equidistant_steps: bool,
                     noret=0):
    """
    Define wave number arrays in Fourier domain used for linear propagation and diffraction
    using the Angular Spectrum method.
    :param control:
    :param equidistant_steps: The flag specifying beam simulation with equidistant steps.
    :param noret: Specifies the wave number operator for the classical Angular Spectrum Method
        of Zemp and Cobbold in regular coordinates (as opposed to retarded time coordinates).
    :return: Full complex wave number operator.
        If control.diffraction_type is set to PseudoDifferential, the wave numbers operator contains
        three layers, the first is the wave numbers in time and eigenvalues of difference matrix A.
        The second layer is the inverse eigenvector matrix Q, and the third the matrix Q.
    """
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t
    mat = control.material.material
    c = mat.sound_speed
    resolution_t = control.signal.resolution_t
    ft = 1 / resolution_t
    df = ft / num_points_t
    kt = 2.0 * numpy.pi / c * numpy.arange(-ft / 2, ft / 2, df)

    if control.diffraction_type == NoDiffraction or \
            control.diffraction_type == ExactDiffraction or \
            control.diffraction_type == AngularSpectrumDiffraction:
        fx = 1 / control.signal.resolution_x
        fy = 1 / control.signal.resolution_y
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
    elif control.diffraction_type == PseudoDifferential:
        raise NotImplementedError
    elif control.diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
            control.diffraction_type == FiniteDifferenceTimeDifferenceFull:
        raise NotImplementedError

    # calculate attenuation if propagation is linear
    loss = numpy.zeros((kt.size))
    if control.attenuation and control.non_linearity is False:
        w = get_frequencies(num_points_t, control.signal.resolution_t / (
                2.0 * numpy.pi * ScaleForTemporalVariable))
        epsa = mat.eps_a
        epsb = mat.eps_b
        loss = epsa * numpy.conj(hilbert(numpy.abs(w) ** epsb)) / ScaleForSpatialVariablesZ

    # assembly of wave-number operator
    if control.diffraction_type == AngularSpectrumDiffraction:
        # assign to vectors
        _wave_numbers = numpy.zeros((numpy.max((num_points_x, num_points_y, num_points_t)), 4))
        _wave_numbers[:num_points_x, 0] = numpy.fft.ifftshift(kx)
        _wave_numbers[:num_points_y, 1] = numpy.fft.ifftshift(ky)
        _wave_numbers[:num_points_t, 2] = numpy.fft.ifftshift(kt)
        _wave_numbers[:num_points_t, 3] = loss
        return _wave_numbers
    else:
        # building full complex wave number operator
        if control.num_dimensions == 3:
            Ky2, Kx2 = numpy.meshgrid(numpy.fft.ifftshift(ky ** 2), numpy.fft.ifftshift(kx ** 2),
                                      indexing='ij')
            Kxy2 = Kx2 + Ky2
            Kxy2 = Kxy2.reshape((num_points_x * num_points_y, 1))
        elif control.num_dimensions == 2:
            Kxy2 = numpy.fft.ifftshift(kx ** 2)
        else:
            Kxy2 = 0
    Kt, KXY = numpy.meshgrid(numpy.fft.ifftshift(kt), Kxy2, indexing='ij')
    _wave_numbers = numpy.sqrt((Kt ** 2 - KXY).astype(complex))
    _wave_numbers = numpy.sign(Kt) * _wave_numbers.real - 1j * _wave_numbers.imag

    # introduces retarded time
    if noret == 0:
        _wave_numbers = _wave_numbers - Kt

    # introduces loss in wave number operator
    if control.attenuation:
        for ii in range(0, num_points_t):
            _wave_numbers[ii, ...] = _wave_numbers[ii, ...] - 1j * loss[ii]

    # convert wave number operator to propagation operator
    if equidistant_steps:
        step_size = control.simulation.step_size
        if control.non_linearity:
            nsubsteps = int(numpy.ceil(step_size / control.signal.resolution_z))
            step_size = step_size / nsubsteps
        _wave_numbers = numpy.exp(-1j * _wave_numbers * step_size)

    if control.diffraction_type == PseudoDifferential:
        raise NotImplementedError

    return _wave_numbers
