import numpy

from material.muscle import Muscle
from propagation.nonlinear.attenuationsolve import attenuationsolve
from propagation.nonlinear.burgerssolve import burgerssolve
from propagation.nonlinear.get_shockdist import get_shockdist


def nonlinattenuationsplit(t,
                           u,
                           dz,
                           shockstep,
                           material=Muscle(37),
                           non_linearity=True,
                           attenuation=True):
    # Initiation of sizes
    num_dimensions = u.ndim
    if num_dimensions == 3:
        nt, ny, nx = u.shape
        u = u.reshape((nt, nx * ny))
    else:
        nt, nx = u.shape
        ny = 1

    # Check material
    isreg = material.is_regular
    epsn = material.eps_n
    epsa = material.eps_a
    epsb = material.eps_b

    for ii in range(nx * ny):
        nstep = 0
        utmp = u[:, ii]
        dztmp = dz
        while dztmp > 0:
            zstep = numpy.minimum(dztmp, shockstep * get_shockdist(t, utmp, epsn))
            dztmp = dztmp - zstep
            nstep = nstep + 1
            if isreg:
                # for regular materials
                if non_linearity and attenuation is False:
                    utmp = burgerssolve(t, utmp, utmp, epsn, zstep)
                elif non_linearity and attenuation:
                    utmp = burgerssolve(t, utmp, utmp, epsn, zstep / 2)
                    utmp = attenuationsolve(t, utmp, zstep, epsa, epsb)
                    utmp = burgerssolve(t, utmp, utmp, epsn, zstep / 2)
                elif non_linearity is False and attenuation:
                    utmp = attenuationsolve(t, utmp, zstep, epsa, epsb)
            else:
                raise NotImplementedError
        u[:, ii] = utmp

    if num_dimensions == 3:
        u = u.reshape((nt, ny, nx))

    return u