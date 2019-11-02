import numpy

from diffraction.diffraction import ExactDiffraction, AngularSpectrumDiffraction, \
    PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from propagation.get_wavenumbers import get_wavenumbers
from propagation.nonlinear.nonlinearpropagate import nonlinearpropagate


def propagate(u_z,
              direction: int,
              control,
              equidistant_steps,
              Kz=None):
    global KZ

    if Kz is not None:
        KZ = Kz
        del Kz

    diffraction_type = control.diffraction_type
    non_linearity = control.non_linearity
    attenuation = control.attenuation
    step_size = control.simulation.step_size

    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t

    # Update position and chose propagation mode
    if direction > 0:
        control.simulation.current_position = control.simulation.current_position + step_size
    elif direction < 0:
        control.simulation.current_position = control.simulation.current_position - step_size

    if (diffraction_type == ExactDiffraction or
        diffraction_type == AngularSpectrumDiffraction or
        diffraction_type == PseudoDifferential) and \
            (non_linearity is False or abs(direction) == 2):
        # Linear propagation and exact diffraction
        if KZ.size == 0:
            KZ = get_wavenumbers(control, equidistant_steps)

        # Forward spatial transform
        if diffraction_type == ExactDiffraction or diffraction_type == AngularSpectrumDiffraction:
            if control.num_dimensions == 3:
                u_z = numpy.fft.fftn(u_z, axes=(2,))
                u_z = numpy.fft.fftn(u_z, axes=(1,))
                u_z = u_z.reshape((num_points_t, num_points_x * num_points_y))
            else:
                u_z = numpy.fft.fftn(u_z, axes=(1,))
        elif diffraction_type == PseudoDifferential:
            tmp = KZ[:, num_points_x:]
            raise NotImplementedError

        # Forward temporal FFT
        u_z = numpy.fft.fftn(u_z, axes=(0,))  # FFT in time

        # Propagation step
        if diffraction_type == ExactDiffraction or diffraction_type == PseudoDifferential:
            if equidistant_steps:
                u_z = u_z * numpy.squeeze(KZ[:num_points_t, :num_points_x * num_points_y])
            else:
                u_z = u_z * numpy.exp(
                    (-1j * step_size) * numpy.squeeze(
                        KZ[:num_points_t, :num_points_x * num_points_y, 0]))
        elif diffraction_type == AngularSpectrumDiffraction:
            raise NotImplementedError
            kx = KZ[:num_points_x, 0]
            ky = KZ[:num_points_y, 1]
            kt = KZ[:num_points_t, 2]
            loss = KZ[:num_points_t, 3]
            kk = 1
            for ii in range(num_points_x):
                kx2 = kx[ii] ** 2
                for jj in range(num_points_y):
                    ky2 = ky[jj] ** 2
                    # Calculate propagation wave numbers
                    kz = numpy.sqrt(kt ** 2 - kx2 - ky2)
                    # Dampen evanescent waves
                    kz = numpy.sign(kt) * kz.real - 1j * kz.imag
                    kz = kz - kt - 1j * loss
                    u_z[:, kk] = u_z[:, kk] * numpy.exp((-1j * step_size) * kz)
                    kk = kk + 1

        # Backward temporal FFT
        u_z = numpy.fft.ifftn(u_z, axes=(0,))

        # Backward spatial transform
        if diffraction_type == ExactDiffraction or diffraction_type == AngularSpectrumDiffraction:
            if control.num_dimensions == 3:
                u_z = u_z.reshape((num_points_t, num_points_y, num_points_x))
                u_z = numpy.fft.ifftn(u_z, axes=(1,))
                u_z = numpy.fft.ifftn(u_z, axes=(2,))
            else:
                u_z = numpy.fft.ifftn(u_z, axes=(1,))
            u_z = u_z.real
        elif diffraction_type == PseudoDifferential:
            u_z = u_z.real
            raise NotImplementedError
        elif (diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
              diffraction_type == FiniteDifferenceTimeDifferenceFull) or \
                (non_linearity or attenuation):
            # Nonlinear propagation in external function
            raise NotImplementedError
    elif (diffraction_type == FiniteDifferenceTimeDifferenceReduced or \
          diffraction_type == FiniteDifferenceTimeDifferenceFull) or \
            (non_linearity or attenuation):
        # Nonlinear propagation in external function
        u_z = nonlinearpropagate(u_z, direction, control, equidistant_steps, KZ)
    else:
        print('Propagation type must be specified')
        exit(-1)

    return u_z
