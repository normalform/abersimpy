import numpy

from misc.timeshift import timeshift
from transducer.get_focal_curvature import get_focal_curvature
from transducer.get_xdidx import get_xdidx


def focus_pulse(u_z,
                prop_control,
                lensfoc=None,
                nofocflag=0,
                physlens=0):
    if lensfoc is None:
        lensfoc = numpy.zeros(2)
        if prop_control.Nex == 1:
            lensfoc[0] = prop_control.Fx
        if prop_control.Ney == 1:
            lensfoc[1] = prop_control.Fy

    # initiate variables
    Fx = prop_control.Fx
    Fy = prop_control.Fy
    Nex = prop_control.Nex
    Ney = prop_control.Ney
    esizex = prop_control.esizex
    esizey = prop_control.esizey
    dx = prop_control.dx
    dy = prop_control.dy
    dt = prop_control.dt
    diffraction_type = prop_control.config.diffraction_type
    annular_transducer = prop_control.config.annular_transducer
    c = prop_control.material.sound_speed

    if isinstance(physlens, int) is False:
        raise NotImplementedError
    physlensx = physlens
    physlensy = physlens

    # find sizes and indices
    (idxxs, idxys, _, _, _) = get_xdidx(prop_control)
    (nt, uny, unx) = u_z.shape
    if unx == 1:
        unx = uny
        uny = 1
    if unx >= numpy.max(idxxs) and uny >= numpy.max(idxys):
        raise NotImplementedError
    else:
        idxxs = range(unx)
        idxys = range(0, uny)
        u_foc = u_z
    xdnx = len(idxxs)
    xdny = len(idxys)

    # calculate focusing
    if annular_transducer:
        raise NotImplementedError
    else:
        # straight forward rectangular transducer
        Rx = get_focal_curvature(Fx, xdnx, Nex, dx, esizex, lensfoc[0], annular_transducer, diffraction_type)
        Ry = get_focal_curvature(Fy, xdny, Ney, dy, esizey, lensfoc[1], annular_transducer, diffraction_type)
        if isinstance(Ry, numpy.ndarray) is False:
            Ry = numpy.array([Ry])
        deltafocx = numpy.ones((xdny, 1)) * numpy.transpose(Rx) / c
        deltafocy = Ry[..., numpy.newaxis] / c * numpy.ones(xdnx)
        if physlensx != 0:
            deltafocx = deltafocx - numpy.max(deltafocx)
        else:
            deltafocx = deltafocx - numpy.min(deltafocx)
        if physlensy != 0:
            deltafocy = deltafocy - numpy.max(deltafocy)
        else:
            deltafocy = deltafocy - numpy.min(deltafocy)
        deltafoc = deltafocx + deltafocy

    if nofocflag == 0:
        u_z[:, slice(idxys[0], idxys[-1] + 1), slice(idxxs[0], idxxs[-1] + 1)] = timeshift(u_foc, -deltafoc / dt, 'fft')

    return u_z, deltafoc
