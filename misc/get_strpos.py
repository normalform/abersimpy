import numpy


def get_strpos(pos):
    pos = numpy.round(100 * pos) / 100
    pos = pos / 100
    n = 6

    num = numpy.zeros(n, dtype=int)
    for ii in range(n):
        num[ii] = int(numpy.floor(pos))
        pos = pos - num[ii]
        pos = 10 * pos

    tmp = num[-1]
    cl = 0
    for ii in range(n-1):
        if num[n-ii - 1] == 9 and tmp == 9:
            cl = cl + 1
            tmp = num[n-ii - 1]
        else:
            tmp = num[n-ii - 1]

    if cl > 1:
        num[n - 1] = 0
        for ii in range(cl):
            num[n - ii - 1] = int(numpy.mod(num[n - ii - 1] + 1, 10))

    strpos = '_{:1d}{:1d}{:1d}{:1d}{:1d}'.format(num[0], num[1], num[2], num[3], num[4])

    return strpos

