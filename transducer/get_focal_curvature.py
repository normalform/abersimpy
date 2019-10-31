import numpy
import scipy.interpolate

from controls.prop_control import NoDiffraction, AngularSpectrumDiffraction, ExactDiffraction, PseudoDifferential


def get_focal_curvature(F,
                        ndx,
                        Nel,
                        resolution_x,
                        elsize=None,
                        lensfoc=numpy.inf,
                        annular_transducer=False,
                        diffraction_type=ExactDiffraction):
    if elsize is None:
        elsize = ndx * resolution_x / Nel

    # if number of points ar one
    if ndx <= 1:
        return 0

    # use delay focusing for more than one element
    if Nel > 1:
        if F == numpy.inf:
            Rd = numpy.zeros((ndx, 1))
        else:
            nsprel = numpy.round(elsize / resolution_x)
            if annular_transducer:
                ae = numpy.arange(0, Nel) * elsize
            else:
                ae = numpy.arange(-int(numpy.floor(Nel / 2)), int(numpy.ceil(Nel / 2))) * elsize + numpy.mod(Nel + 1,
                                                                                                             2) * elsize / 2

            a = numpy.zeros((ae.size * int(nsprel)))
            for x in range(int(nsprel)):
                a[x::int(nsprel)] = ae
            if annular_transducer:
                if numpy.mod(nsprel, 2) == 0:
                    a = a[int(numpy.ceil(nsprel / 2)):] + resolution_x / 2
                else:
                    a = a[int(numpy.floor(nsprel / 2)):]
            if diffraction_type == NoDiffraction or \
                    diffraction_type == ExactDiffraction or \
                    diffraction_type == AngularSpectrumDiffraction or \
                    diffraction_type == PseudoDifferential:
                Rd = numpy.sqrt(a ** 2 + F ** 2) - F
            else:
                Rd = a ** 2 / (2 * F)
    else:
        Rd = numpy.zeros(ndx)

    x = numpy.arange(numpy.max(Rd.shape))
    xi = numpy.linspace(0, numpy.max(Rd.shape) - 1, ndx)
    intpf = scipy.interpolate.interp1d(numpy.transpose(x), Rd, kind='nearest')
    Rd = intpf(numpy.transpose(xi))

    # use evnetual lens focusing
    if lensfoc is not numpy.inf and lensfoc != 0.0:
        if annular_transducer:
            ac = numpy.arange(0, ndx) * resolution_x + numpy.mod(ndx + 1, 2) * resolution_x / 2
        else:
            ac = numpy.arange(-int(numpy.floor(ndx / 2)), int(numpy.ceil(ndx / 2))) * resolution_x + numpy.mod(ndx + 1,
                                                                                                               2) * resolution_x / 2
        if diffraction_type == NoDiffraction or \
                diffraction_type == ExactDiffraction or \
                diffraction_type == AngularSpectrumDiffraction or \
                diffraction_type == PseudoDifferential:
            R1 = numpy.sqrt(ac ** 2 + lensfoc ** 2) - lensfoc
        else:
            R1 = ac ** 2 / (2 * lensfoc)
    else:
        R1 = 0

    R = Rd + R1
    R = R - numpy.min(R)

    return R
