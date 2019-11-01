import numpy


def get_xdidx(control):
    # initiation of parameters
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    resolution_x = control.signal.resolution_x
    resolution_y = control.signal.resolution_y
    nex = control.transducer.num_elements_azimuth
    ney = control.transducer.num_elements_elevation
    elements_size_azimuth = control.transducer.elements_size_azimuth
    elements_size_elevation = control.transducer.elements_size_elevation
    cc = control.transducer.center_channel
    num_dimensions = control.num_dimensions
    annular_transducer = control.annular_transducer

    # set lengths of transducers
    if num_dimensions == 1:
        idxxs = 1
        idxys = 1
        idxx0 = []
        idxy0 = []
        ccs = [1, 1]
        return idxxs, idxys, idxx0, idxy0, ccs

    if annular_transducer and num_dimensions == 2:
        ndx = numpy.round((2 * nex - 1) * elements_size_azimuth / resolution_x)
    elif annular_transducer:
        ndx = numpy.round((2 * nex - 1) * elements_size_azimuth / resolution_x)
        ndy = numpy.round((2 * ney - 1) * elements_size_elevation / resolution_y)
    else:
        ndx = numpy.round(nex * elements_size_azimuth / resolution_x)
        ndy = numpy.round(ney * elements_size_elevation / resolution_y)

    if num_dimensions == 2:
        ndy = 1

    # set up indices
    if annular_transducer and num_dimensions == 2:
        idxxs = numpy.arange(0, numpy.ceil(ndx / 2)) + cc[0]
        idxys = numpy.array(1)
    else:
        idxxs = numpy.arange(-numpy.floor(ndx / 2), numpy.ceil(ndx / 2)) + cc[0]
        idxys = numpy.arange(-numpy.floor(ndy / 2), numpy.ceil(ndy / 2)) + cc[1]

    idxx0 = numpy.setxor1d(idxxs, numpy.arange(1, num_points_x + 1))
    idxy0 = numpy.setxor1d(idxys, numpy.arange(1, num_points_y + 1))

    # set center index for transducer field
    ccs = numpy.zeros((2, 1))
    if annular_transducer and num_dimensions == 2:
        ccs[0] = 1
        ccs[1] = 1
    else:
        ccs[0] = numpy.floor(ndx / 2) + 1
        ccs[1] = numpy.floor(ndy / 2) + 1
        if cc[1] <= 1:
            cc[1] = 1

    return idxxs.astype(int) - 1, idxys.astype(int) - 1, idxx0.astype(int) - 1, idxy0.astype(int) - 1, ccs.astype(
        int) - 1
