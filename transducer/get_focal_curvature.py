"""
get_focal_curvature.py
"""
import numpy
import scipy.interpolate

from diffraction.diffraction import NoDiffraction, AngularSpectrumDiffraction, ExactDiffraction, \
    PseudoDifferential


def get_focal_curvature(focal_depth,
                        num_points,
                        num_elements,
                        resolution_x,
                        element_size=None,
                        lens_focal_depth=numpy.inf,
                        annular_transducer=False,
                        diffraction_type=ExactDiffraction):
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
        return 0

    # use delay focusing for more than one element
    if num_elements > 1:
        if focal_depth == numpy.inf:
            _rd = numpy.zeros((num_points, 1))
        else:
            _nsprel = numpy.round(element_size / resolution_x)
            if annular_transducer:
                _ae = numpy.arange(0, num_elements) * element_size
            else:
                _ae = numpy.arange(-int(numpy.floor(num_elements / 2)),
                                   int(numpy.ceil(num_elements / 2))) * \
                      element_size + numpy.mod(num_elements + 1, 2) * element_size / 2
            _a = numpy.zeros((_ae.size * int(_nsprel)))
            for _x in range(int(_nsprel)):
                _a[_x::int(_nsprel)] = _ae

            if annular_transducer:
                if numpy.mod(_nsprel, 2) == 0:
                    _a = _a[int(numpy.ceil(_nsprel / 2)):] + resolution_x / 2
                else:
                    _a = _a[int(numpy.floor(_nsprel / 2)):]
            if diffraction_type in (NoDiffraction,
                                    ExactDiffraction,
                                    AngularSpectrumDiffraction,
                                    PseudoDifferential):
                _rd = numpy.sqrt(_a ** 2 + focal_depth ** 2) - focal_depth
            else:
                _rd = _a ** 2 / (2 * focal_depth)
    else:
        _rd = numpy.zeros(num_points)

    _x = numpy.arange(numpy.max(_rd.shape))
    _xi = numpy.linspace(0, numpy.max(_rd.shape) - 1, num_points)
    _interpolation_function = scipy.interpolate.interp1d(numpy.transpose(_x), _rd, kind='nearest')
    _rd = _interpolation_function(numpy.transpose(_xi))

    # use eventual lens focusing
    if lens_focal_depth is not numpy.inf and lens_focal_depth != 0.0:
        if annular_transducer:
            _ac = numpy.arange(0, num_points) * \
                  resolution_x + numpy.mod(num_points + 1, 2) * resolution_x / 2
        else:
            _ac = numpy.arange(-int(numpy.floor(num_points / 2)),
                               int(numpy.ceil(num_points / 2))) * \
                  resolution_x + numpy.mod(num_points + 1, 2) * resolution_x / 2
        if diffraction_type in (NoDiffraction,
                                ExactDiffraction,
                                AngularSpectrumDiffraction,
                                PseudoDifferential):
            _r1 = numpy.sqrt(_ac ** 2 + lens_focal_depth ** 2) - lens_focal_depth
        else:
            _r1 = _ac ** 2 / (2 * lens_focal_depth)
    else:
        _r1 = 0

    _r = _rd + _r1
    _r = _r - numpy.min(_r)

    return _r
