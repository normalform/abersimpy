# -*- coding: utf-8 -*-
"""
    get_focal_curvature.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import math
from typing import Optional

import numpy
import scipy.interpolate

from system.diffraction.diffraction import NoDiffraction, AngularSpectrumDiffraction, \
    ExactDiffraction, PseudoDifferential
from system.diffraction.interfaces import IDiffractionType


def get_focal_curvature(focal_depth,
                        num_points,
                        num_elements,
                        resolution_x,
                        element_size: Optional[float] = None,
                        lens_focal_depth: Optional[float] = math.inf,
                        annular_transducer: Optional[bool] = False,
                        diffraction_type: Optional[IDiffractionType] = ExactDiffraction) \
        -> numpy.ndarray:
    """
    Calculates focal curvature.
    :param focal_depth: Focal depth for delay focusing.
    :param num_points: Number of points across transducer.
    :param num_elements: Number of elements.
    :param resolution_x: Resolution across transducer.
    :param element_size: Element size of transducer.
    :param lens_focal_depth: Focal depth for lens focusing. Default is Inf;
    :param annular_transducer: Flag for annular transducer.
    :param diffraction_type: Flag specifying diffraction model. If diffraction_type is one of
        NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction or, PseudoDifferential,
        a full wave equation and spherical focal curvature is assumed.
        For diffraction_type is one of FiniteDifferenceTimeDifferenceReduced or
        FiniteDifferenceTimeDifferenceFull, a parabolic approximation is assumed.
    :return:
    """
    if element_size is None:
        _element_size = num_points * resolution_x / num_elements
    else:
        _element_size = element_size

    # if number of points is one
    if num_points <= 1:
        return numpy.array(0, dtype=float)

    # use delay focusing for more than one element
    if num_elements > 1:
        if focal_depth == math.inf:
            rd = numpy.zeros((num_points, 1))
        else:
            nsprel = numpy.round(element_size / resolution_x)
            if annular_transducer:
                ae = numpy.arange(0, num_elements) * element_size
            else:
                ae = numpy.arange(-int(numpy.floor(num_elements / 2)),
                                  int(numpy.ceil(num_elements / 2))) * \
                     element_size + numpy.mod(num_elements + 1, 2) * element_size / 2
            a = numpy.zeros((ae.size * int(nsprel)))
            for x in range(int(nsprel)):
                a[x::int(nsprel)] = ae

            if annular_transducer:
                if numpy.mod(nsprel, 2) == 0:
                    a = a[int(numpy.ceil(nsprel / 2)):] + resolution_x / 2
                else:
                    a = a[int(numpy.floor(nsprel / 2)):]
            if diffraction_type in (NoDiffraction,
                                    ExactDiffraction,
                                    AngularSpectrumDiffraction,
                                    PseudoDifferential):
                rd = numpy.sqrt(a ** 2 + focal_depth ** 2) - focal_depth
            else:
                rd = a ** 2 / (2 * focal_depth)
    else:
        rd = numpy.zeros(num_points)

    x = numpy.arange(numpy.max(rd.shape))
    xi = numpy.linspace(0, numpy.max(rd.shape) - 1, num_points)
    interpolation_function = scipy.interpolate.interp1d(x.T, rd, kind='nearest')
    rd = interpolation_function(xi.T)

    # use eventual lens focusing
    if lens_focal_depth is not math.inf and lens_focal_depth != 0.0:
        if annular_transducer:
            ac = numpy.arange(0, num_points) * \
                 resolution_x + numpy.mod(num_points + 1, 2) * resolution_x / 2
        else:
            ac = numpy.arange(-int(numpy.floor(num_points / 2)),
                              int(numpy.ceil(num_points / 2))) * \
                 resolution_x + numpy.mod(num_points + 1, 2) * resolution_x / 2
        if diffraction_type in (NoDiffraction,
                                ExactDiffraction,
                                AngularSpectrumDiffraction,
                                PseudoDifferential):
            r1 = numpy.sqrt(ac ** 2 + lens_focal_depth ** 2) - lens_focal_depth
        else:
            r1 = ac ** 2 / (2 * lens_focal_depth)
    else:
        r1 = 0

    r = rd + r1
    r = r - numpy.min(r)

    return r
