import numpy

from misc.raised_cos import raised_cos


def get_window(N = (256, 1),
               ds = None,
               Lw = 0.1,
               L0 = 0.1,
               annular_transducer = 0):
    if ds is None:
        ds = 1 / N[0]

    if isinstance(N, int):
        N = (N, 1)
    if N[0] == 1 and N[1] == 1:
        return -1

    # return rectangular window
    if annular_transducer == 2:
        w = numpy.ones(N[0]*N[1])
        return w

    # create two element vectors
    if isinstance(ds, float):
        ds = (ds, ds)
    if isinstance(Lw, float):
        Lw = (Lw, Lw)
    if isinstance(L0, float):
        L0 = (L0, L0)

    # calculate number of points of fall-off and zero part
    nLwx = int(numpy.ceil(Lw[0] / ds[0]))
    nL0x = int(numpy.ceil(L0[0] / ds[0]))

    # adjust Nx for annular_transducer = 1
    Ny = N[1]
    if Ny == 1 and annular_transducer:
        Nx = 2 * N[0]
        nLwy = 0
        nL0y = 0
    else:
        Nx = N[0]
        nLwy = int(numpy.ceil(Lw[1] / ds[1]))
        nL0y = int(numpy.ceil(L0[1] / ds[1]))

    # find length of window
    nwx = Nx - 2 * nL0x
    nwy = Ny - 2 * nL0y

    # create window in x
    wx = raised_cos(nwx, nLwx, Nx)
    if Ny == 1 and annular_transducer:
        wx = wx[N[0]:]
        Nx = Nx / 2

    # create window in y
    if Ny == 1:
        wy = numpy.array([1])
    else:
        wy = raised_cos(nwy, nLwy, Ny)

    # reshape window
    w = wy[..., numpy.newaxis] * numpy.transpose(wx)
    w = w.reshape(Nx * Ny)

    return w
