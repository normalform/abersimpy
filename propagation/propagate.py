import numpy

from prop_control import ExactDiffraction, AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from propagation.get_wavenumbers import get_wavenumbers
from propagation.nonlinear.nonlinearpropagate import nonlinearpropagate


def propagate(u_z,
              dir,
              prop_control,
              Kz=None):
    global KZ

    if Kz is not None:
        KZ = Kz
        del Kz

    diffraction_type = prop_control.config.diffraction_type
    non_linearity = prop_control.config.non_linearity
    attenuation = prop_control.config.attenuation
    step_size = prop_control.step_size

    num_points_x = prop_control.num_points_x
    num_points_y = prop_control.num_points_y
    num_points_t = prop_control.num_points_t

    # Update position and chose propagation mode
    if dir > 0:
        prop_control.current_position = prop_control.current_position + step_size
    elif dir < 0:
        prop_control.current_position = prop_control.current_position - step_size

    if (diffraction_type == ExactDiffraction or
        diffraction_type == AngularSpectrumDiffraction or
        diffraction_type == PseudoDifferential) and \
            (non_linearity is False or abs(dir) == 2):
        # Linear propagation and exact diffraction
        if KZ.size == 0:
            KZ = get_wavenumbers(prop_control)

        # Forward spatial transform
        if diffraction_type == ExactDiffraction or diffraction_type == AngularSpectrumDiffraction:
            if prop_control.num_dimensions == 3:
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
            if prop_control.config.equidistant_steps:
                u_z = u_z * numpy.squeeze(KZ[:num_points_t, :num_points_x * num_points_y])
            else:
                u_z = u_z * numpy.exp(
                    (-1j * step_size) * numpy.squeeze(KZ[:num_points_t, :num_points_x * num_points_y, 0]))
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
            if prop_control.num_dimensions == 3:
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
        u_z = nonlinearpropagate(u_z, dir, prop_control, KZ)
    else:
        print('Propagation type must be specified')
        exit(-1)

    return u_z, prop_control
