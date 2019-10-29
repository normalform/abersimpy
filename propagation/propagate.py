from initpropcontrol import initpropcontrol
from propagation.get_wavenumbers import get_wavenumbers
from propagation.nonlinear.nonlinearpropagate import nonlinearpropagate
from propcontrol import ExactDiffraction, AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull

import numpy


def propagate(u_z,
              dir,
              propcontrol=None,
              Kz=None):
    global KZ

    if propcontrol is None:
        propcontrol = initpropcontrol();
    if Kz is not None:
        KZ = Kz
        del Kz

    diffraction_type = propcontrol.config.diffraction_type
    non_linearity = propcontrol.config.non_linearity
    attenuation = propcontrol.config.attenuation
    stepsize = propcontrol.stepsize

    nx = propcontrol.nx
    ny = propcontrol.ny
    nt = propcontrol.nt

    # Update position and chose propagation mode
    if dir > 0:
        propcontrol.currentpos = propcontrol.currentpos + stepsize
    elif dir < 0:
        propcontrol.currentpos = propcontrol.currentpos - stepsize

    if (diffraction_type == ExactDiffraction or
        diffraction_type == AngularSpectrumDiffraction or
        diffraction_type == PseudoDifferential) and \
            (non_linearity is False or abs(dir) == 2):
        # Linear propagation and exact diffraction
        if KZ.size == 0:
            KZ = get_wavenumbers(propcontrol)

        # Forward spatial transform
        if diffraction_type == ExactDiffraction or diffraction_type == AngularSpectrumDiffraction:
            if propcontrol.num_dimensions == 3:
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
            if propcontrol.config.equidistant_steps:
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
            if propcontrol.num_dimensions == 3:
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
        u_z = nonlinearpropagate(u_z, dir, propcontrol, KZ)
    else:
        print('Propagation type must be specified')
        exit(-1)

    return u_z, propcontrol
