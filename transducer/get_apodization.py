import numpy
import scipy.signal


def get_apodization(num_points_x,
                    num_points_y=1,
                    type='tukey',
                    s=0,
                    annular_transducer=False):
    # return apodization based on type
    if num_points_y == 1 and annular_transducer is False and isinstance(s, list) is False:
        if type == 'tukey':
            apod = scipy.signal.tukey(num_points_x, s)
        elif type == 'hamming':
            apod = numpy.hamming(num_points_x)
        elif type == 'hanning':
            apod = numpy.hanning(num_points_x)
        elif type == 'hann':
            apod = scipy.signal.hann(num_points_x)
        elif type == 'rect':
            apod = scipy.signal.boxcar(num_points_x)
        return apod

    # calculate 2d and annular apodizations
    if annular_transducer:
        if num_points_y == 1:
            # get apodization for axis-symmetric simulations
            apodx = get_apodization(2 * num_points_x - 1, 1, type, s[0])
            apod = apodx[num_points_x:]
        elif num_points_x == num_points_y:
            # get apodization for annual transducer in full 3D
            raise NotImplementedError
        else:
            # return rectangular apodization
            print('For annular transducers, num_points_x and num_points_y has to be the same')
            apod = numpy.ones((num_points_y, num_points_x))
            return apod
    else:
        if len(s) == 2:
            # different windows in x and y
            apodx = get_apodization(num_points_x, 1, type, s[0])
            apody = get_apodization(num_points_y, 1, type, s[1])
        else:
            # equal windows in x and y
            apodx = get_apodization(num_points_x, 1, type, s[0])
            apody = get_apodization(num_points_y, 1, type, s[1])

        apod = apody[..., numpy.newaxis] * numpy.transpose(apodx)[numpy.newaxis, ...]
        return apod
