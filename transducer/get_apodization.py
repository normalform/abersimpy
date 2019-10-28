import numpy
import scipy.signal

import matplotlib.pyplot as plt


def get_apodization(nx,
                    ny = 1,
                    type = 'tukey',
                    s = 0,
                    annflag = 0):
    # return apodization based on type
    if ny == 1 and annflag == 0 and isinstance(s, list) is False:
        if type == 'tukey':
            apod = scipy.signal.tukey(nx, s)
        elif type == 'hamming':
            apod = numpy.hamming(nx)
        elif type == 'hanning':
            apod = numpy.hanning(nx)
        elif type == 'hann':
            apod = scipy.signal.hann(nx)
        elif type == 'rect':
            apod = scipy.signal.boxcar(nx)
        return apod

    # calculate 2d and annular apodizations
    if annflag != 0:
        if ny == 1:
            # get apodization for axis-symmetric simulations
            apodx = get_apodization(2 * nx - 1, 1, type, s[0])
            apod = apodx[nx:]
        elif nx == ny:
            # get apodization for annual transducer in full 3D
            raise NotImplementedError
        else:
            # return rectangular apodization
            print('For annular transducers, nx and ny has to be the same')
            apod = numpy.ones((ny, nx))
            return apod
    else:
        if len(s) == 2:
            # different windows in x and y
            apodx = get_apodization(nx, 1, type, s[0])
            apody = get_apodization(ny, 1, type, s[1])
        else:
            # equal windows in x and y
            apodx = get_apodization(nx, 1, type, s[0])
            apody = get_apodization(ny, 1, type, s[1])

        apod = apody[..., numpy.newaxis] * numpy.transpose(apodx)[numpy.newaxis, ...]
        return apod

