import numpy

from prop_control import PropControl, ExactDiffraction, AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from propagation.get_wavenumbers import get_wavenumbers
from propagation.nonlinear.nonlinearpropagate import nonlinearpropagate


def propagate(u_z,
              dir,
              prop_control=None,
              Kz=None):
    global KZ

    if prop_control is None:
        prop_control = PropControl.init_prop_control();
    if Kz is not None:
        KZ = Kz
        del Kz

    diffraction_type = prop_control.config.diffraction_type
    non_linearity = prop_control.config.non_linearity
    attenuation = prop_control.config.attenuation
    stepsize = prop_control.stepsize

    nx = prop_control.nx
    ny = prop_control.ny
    nt = prop_control.nt

    # Update position and chose propagation mode
    if dir > 0:
        prop_control.currentpos = prop_control.currentpos + stepsize
    elif dir < 0:
        prop_control.currentpos = prop_control.currentpos - stepsize

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
                u_z = u_z.reshape((nt, nx * ny))
            else:
                u_z = numpy.fft.fftn(u_z, axes=(1,))
        elif diffraction_type == PseudoDifferential:
            tmp = KZ[:, nx:]
            raise NotImplementedError

        # Forward temporal FFT
        u_z = numpy.fft.fftn(u_z, axes=(0,))  # FFT in time

        # Propagation step
        if diffraction_type == ExactDiffraction or diffraction_type == PseudoDifferential:
            if prop_control.config.equidistant_steps:
                u_z = u_z * numpy.squeeze(KZ[:nt, :nx * ny])
            else:
                u_z = u_z * numpy.exp((-1j * stepsize) * numpy.squeeze(KZ[:nt, :nx * ny, 0]))
        elif diffraction_type == AngularSpectrumDiffraction:
            raise NotImplementedError
            kx = KZ[:nx, 0]
            ky = KZ[:ny, 1]
            kt = KZ[:nt, 2]
            loss = KZ[:nt, 3]
            kk = 1
            for ii in range(nx):
                kx2 = kx[ii] ** 2
                for jj in range(ny):
                    ky2 = ky[jj] ** 2
                    # Calculate propagation wave numbers
                    kz = numpy.sqrt(kt ** 2 - kx2 - ky2)
                    # Dampen evanescent waves
                    kz = numpy.sign(kt) * kz.real - 1j * kz.imag
                    kz = kz - kt - 1j * loss
                    u_z[:, kk] = u_z[:, kk] * numpy.exp((-1j * stepsize) * kz)
                    kk = kk + 1

        # Backward temporal FFT
        u_z = numpy.fft.ifftn(u_z, axes=(0,))

        # Backward spatial transform
        if diffraction_type == ExactDiffraction or diffraction_type == AngularSpectrumDiffraction:
            if prop_control.num_dimensions == 3:
                u_z = u_z.reshape((nt, ny, nx))
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
