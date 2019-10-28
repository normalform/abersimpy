from initpropcontrol import initpropcontrol
from propagation.get_wavenumbers import get_wavenumbers
from propagation.nonlinear.nonlinearpropagate import nonlinearpropagate

import numpy
import matplotlib.pyplot as plt


def propagate(u_z,
              dir,
              propcontrol = None,
              Kz = None):
    global KZ

    if propcontrol is None:
        propcontrol = initpropcontrol();
    if Kz is not None:
        KZ = Kz
        del Kz

    diffrflag = propcontrol.diffrflag
    nonlinflag = propcontrol.nonlinflag
    lossflag = propcontrol.lossflag
    stepsize = propcontrol.stepsize

    nx = propcontrol.nx
    ny = propcontrol.ny
    nt = propcontrol.nt

    # Update position and chose propagation mode
    if dir > 0:
        propcontrol.currentpos = propcontrol.currentpos + stepsize
    elif dir < 0:
        propcontrol.currentpos = propcontrol.currentpos - stepsize

    if (diffrflag <= 3 and diffrflag > 0) and (nonlinflag == 0 or abs(dir) == 2):
        # Linear propagation and exact diffraction
        if KZ.size == 0:
            KZ = get_wavenumbers(propcontrol)

        # Forward spatial transform
        if diffrflag == 1 or diffrflag == 2:
            if propcontrol.ndims == 3:
                u_z = numpy.fft.fftn(u_z, axes=(2,))
                u_z = numpy.fft.fftn(u_z, axes=(1,))
                u_z = u_z.reshape((nt, nx * ny))
            else:
                u_z = numpy.fft.fftn(u_z, axes=(1,))
        elif diffrflag == 3:
            tmp = KZ[:,nx:]
            raise NotImplementedError

        # Forward temporal FFT
        u_z = numpy.fft.fftn(u_z, axes=(0,))  # FFT in time

        # Propagation step
        if diffrflag == 1 or diffrflag == 3:
            if propcontrol.equidistflag != 0:
                u_z = u_z * numpy.squeeze(KZ[:nt, :nx*ny])
            else:
                u_z = u_z * numpy.exp((-1j * stepsize) * numpy.squeeze(KZ[:nt, :nx * ny, 0]))
        elif diffrflag == 2:
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
        if diffrflag == 1 or diffrflag == 2:
            if propcontrol.ndims == 3:
                u_z = u_z.reshape((nt, ny, nx))
                u_z = numpy.fft.ifftn(u_z, axes=(1,))
                u_z = numpy.fft.ifftn(u_z, axes=(2,))
            else:
                u_z = numpy.fft.ifftn(u_z, axes=(1,))
            u_z = u_z.real
        elif diffrflag == 3:
            u_z = u_z.real
            raise NotImplementedError
        elif diffrflag > 3 or (nonlinflag != 0 or lossflag != 0):
            # Nonlinear propagation in external function
            raise NotImplementedError
    elif diffrflag > 3 or (nonlinflag != 0 or lossflag != 0):
        # Nonlinear propagation in external function
        u_z = nonlinearpropagate(u_z, dir, propcontrol, KZ)
    else:
        print('Propagation type must be specified')
        exit(-1)

    return u_z, propcontrol