import numpy


def get_diffstencil(numorder=4,
                    difforder=2):
    if difforder == 1:
        if numorder == 2:
            d = numpy.array([-1.0, 0.0, 1.0]) / 2.0
        elif numorder == 4:
            d = numpy.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
        else:
            print('Unrecognized numerical order. Using fourth order differencing')
            d = get_diffstencil(difforder, 4)
    elif difforder == 2:
        if numorder == 2:
            d = numpy.array([-1.0, 2.0, -1.0])
        elif numorder == 4:
            d = numpy.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
        else:
            print('Unrecognized numerical order. Using fourth order differencing')
            d = get_diffstencil(difforder, 4)
    else:
        print('Unrecognized differential order, Using second order differential')
        d = get_diffstencil(2, numorder)

    return d
