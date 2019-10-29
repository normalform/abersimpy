import numpy


def get_xdidx(propcontrol):
    # initiation of parameters
    nx = propcontrol.nx
    ny = propcontrol.ny
    dx = propcontrol.dx
    dy = propcontrol.dy
    nex = propcontrol.Nex
    ney = propcontrol.Ney
    esizex = propcontrol.esizex
    esizey = propcontrol.esizey
    cc = propcontrol.cchannel
    num_dimensions = propcontrol.num_dimensions
    annular_transducer = propcontrol.config.annular_transducer

    # set lengths of transducers
    if num_dimensions == 1:
        idxxs = 1
        idxys = 1
        idxx0 = []
        idxy0 = []
        ccs = [1, 1]
        return idxxs, idxys, idxx0, idxy0, ccs

    if annular_transducer and num_dimensions == 2:
        ndx = numpy.round((2 * nex - 1) * esizex / dx)
    elif annular_transducer:
        ndx = numpy.round((2 * nex - 1) * esizex / dx)
        ndy = numpy.round((2 * ney - 1) * esizey / dy)
    else:
        ndx = numpy.round(nex * esizex / dx)
        ndy = numpy.round(ney * esizey / dy)

    if num_dimensions == 2:
        ndy = 1

    # set up indices
    if annular_transducer and num_dimensions == 2:
        idxxs = numpy.arange(0, numpy.ceil(ndx / 2)) + cc[0]
        idxys = numpy.array(1)
    else:
        idxxs = numpy.arange(-numpy.floor(ndx/2), numpy.ceil(ndx/2)) + cc[0]
        idxys = numpy.arange(-numpy.floor(ndy/2), numpy.ceil(ndy/2)) + cc[1]

    idxx0 = numpy.setxor1d(idxxs, numpy.arange(1, nx+1))
    idxy0 = numpy.setxor1d(idxys, numpy.arange(1, ny+1))

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

    return idxxs.astype(int) - 1, idxys.astype(int) - 1, idxx0.astype(int) - 1, idxy0.astype(int) - 1, ccs.astype(int) - 1
