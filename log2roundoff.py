import numpy


def log2roundoff(np):
    if np >= 0 and np <=2:
        npn = np
    else:
        e = numpy.ceil(numpy.log2(np))
        f = np / 2**e
        npn = 2**e
        if np / 2 ** (e-1) < 1.1:
            npn = 2 ** (e-1)
        else:
            if np < 2**(e-2) * 3:
                npn = 2 ** (e-2) * 3
    return int(npn)
