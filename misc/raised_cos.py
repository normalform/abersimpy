import numpy


def raised_cos(winlength,
               taplength,
               totlength=None):
    winlength = int(numpy.floor(winlength))
    if totlength is None:
        totlength = winlength
    if totlength < winlength:
        totlength = winlength

    # initialize window
    w = numpy.ones(winlength)

    # calculate tapering
    if taplength > 1:
        xtap = numpy.arange(0, taplength) / (taplength - 1)
        w[:taplength] = (1 - numpy.cos(numpy.pi * xtap)) / 2.0
        w[winlength - taplength:winlength] = numpy.flipud(w[:taplength])
    else:
        w[0] = 0
        w[-1] = 0

    # calculate zero-pad region
    l1 = int(numpy.floor((totlength - winlength) / 2))
    l2 = totlength - l1 - winlength

    # zero-pad window
    if l1 > 0:
        w = numpy.concatenate((numpy.zeros(l1), w))
    if l2 > 0:
        w = numpy.concatenate((w, numpy.zeros(l2)))

    return w
