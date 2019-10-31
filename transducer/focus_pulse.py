import numpy

from misc.timeshift import timeshift
from transducer.get_focal_curvature import get_focal_curvature
from transducer.get_xdidx import get_xdidx


def focus_pulse(u_z,
                main_control,
                lensfoc=None,
                nofocflag=0,
                physlens=0):
    if lensfoc is None:
        lensfoc = numpy.zeros(2)
        if main_control.num_elements_azimuth == 1:
            lensfoc[0] = main_control.focus_azimuth
        if main_control.num_elements_elevation == 1:
            lensfoc[1] = main_control.focus_elevation

    # initiate variables
    focus_azimuth = main_control.focus_azimuth
    focus_elevation = main_control.focus_elevation
    num_elements_azimuth = main_control.num_elements_azimuth
    num_elements_elevation = main_control.num_elements_elevation
    elements_size_azimuth = main_control.elements_size_azimuth
    elements_size_elevation = main_control.elements_size_elevation
    resolution_x = main_control.resolution_x
    resolution_y = main_control.resolution_y
    resolution_t = main_control.resolution_t
    diffraction_type = main_control.config.diffraction_type
    annular_transducer = main_control.config.annular_transducer
    c = main_control.material.sound_speed

    if isinstance(physlens, int) is False:
        raise NotImplementedError
    physlensx = physlens
    physlensy = physlens

    # find sizes and indices
    (idxxs, idxys, _, _, _) = get_xdidx(main_control)
    (num_points_t, uny, unx) = u_z.shape
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
        Rx = get_focal_curvature(focus_azimuth,
                                 xdnx,
                                 num_elements_azimuth,
                                 resolution_x,
                                 elements_size_azimuth,
                                 lensfoc[0],
                                 annular_transducer,
                                 diffraction_type)
        Ry = get_focal_curvature(focus_elevation,
                                 xdny,
                                 num_elements_elevation,
                                 resolution_y,
                                 elements_size_elevation,
                                 lensfoc[1],
                                 annular_transducer,
                                 diffraction_type)
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
        u_z[:, slice(idxys[0], idxys[-1] + 1), slice(idxxs[0], idxxs[-1] + 1)] = timeshift(u_foc,
                                                                                           -deltafoc / resolution_t,
                                                                                           'fft')

    return u_z, deltafoc
