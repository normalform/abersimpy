"""
focus_pulse.py
"""
import numpy

from misc.time_shift import time_shift
from transducer.get_focal_curvature import get_focal_curvature
from transducer.get_xdidx import get_xdidx


def focus_pulse(control,
                wave_field,
                lens_focusing=None,
                no_focusing: bool = False,
                physical_lens=0):
    """
    Function that focuses a wave field according to the control.
    :param control: The control.
    :param wave_field: The wave field.
    :param lens_focusing: Additional lens focusing.
    :param no_focusing: If no_focusing is True, the field is not focused,
        but the focusing delays are calculated and returned.
    :param physical_lens: A flag specifying the use of a physical lens, that is a lens where
        the endpoints of the lens are un-altered, and the focusing delays are introduces
        for the interior points. An unphysical lens are adjusted such that the center channel
        does not move during focusing. If physical_lens has two numbers, they are associated
        with the size of the lens.
    :return: Focused wave field, Focal delays
    """
    if lens_focusing is None:
        _lens_focusing = numpy.zeros(2)
        if control.transducer.num_elements_azimuth == 1:
            _lens_focusing[0] = control.transducer.focus_azimuth
        if control.transducer.num_elements_elevation == 1:
            _lens_focusing[1] = control.transducer.focus_elevation
    else:
        _lens_focusing = lens_focusing

    # initiate variables
    _focus_azimuth = control.transducer.focus_azimuth
    _focus_elevation = control.transducer.focus_elevation
    _num_elements_azimuth = control.transducer.num_elements_azimuth
    _num_elements_elevation = control.transducer.num_elements_elevation
    _elements_size_azimuth = control.transducer.elements_size_azimuth
    _elements_size_elevation = control.transducer.elements_size_elevation
    _resolution_x = control.signal.resolution_x
    _resolution_y = control.signal.resolution_y
    _resolution_t = control.signal.resolution_t
    _diffraction_type = control.diffraction_type
    _annular_transducer = control.annular_transducer
    _sound_speed = control.material.material.sound_speed

    if isinstance(physical_lens, int) is False:
        raise NotImplementedError
    _physical_lens_x = physical_lens
    _physical_lens_y = physical_lens

    # find sizes and indices
    idxxs, idxys, _, _, _ = get_xdidx(control)
    (num_points_t, uny, unx) = wave_field.shape
    if unx == 1:
        unx = uny
        uny = 1
    if unx >= numpy.max(idxxs) and uny >= numpy.max(idxys):
        raise NotImplementedError
    else:
        idxxs = range(unx)
        idxys = range(0, uny)
        u_foc = wave_field
    xdnx = len(idxxs)
    xdny = len(idxys)

    # calculate focusing
    if _annular_transducer:
        raise NotImplementedError
    else:
        # straight forward rectangular transducer
        Rx = get_focal_curvature(_focus_azimuth,
                                 xdnx,
                                 _num_elements_azimuth,
                                 _resolution_x,
                                 _elements_size_azimuth,
                                 _lens_focusing[0],
                                 _annular_transducer,
                                 _diffraction_type)
        Ry = get_focal_curvature(_focus_elevation,
                                 xdny,
                                 _num_elements_elevation,
                                 _resolution_y,
                                 _elements_size_elevation,
                                 _lens_focusing[1],
                                 _annular_transducer,
                                 _diffraction_type)
        if isinstance(Ry, numpy.ndarray) is False:
            Ry = numpy.array([Ry])
        deltafocx = numpy.ones((xdny, 1)) * numpy.transpose(Rx) / _sound_speed
        deltafocy = Ry[..., numpy.newaxis] / _sound_speed * numpy.ones(xdnx)
        if _physical_lens_x != 0:
            deltafocx = deltafocx - numpy.max(deltafocx)
        else:
            deltafocx = deltafocx - numpy.min(deltafocx)
        if _physical_lens_y != 0:
            deltafocy = deltafocy - numpy.max(deltafocy)
        else:
            deltafocy = deltafocy - numpy.min(deltafocy)
        deltafoc = deltafocx + deltafocy

    if no_focusing is False:
        wave_field[:, slice(idxys[0], idxys[-1] + 1), slice(idxxs[0], idxxs[-1] + 1)] = \
            time_shift(u_foc, -deltafoc / _resolution_t, 'fft')

    return wave_field, deltafoc
