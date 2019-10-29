import numpy


def makedb(u,
           dyn = None,
           nrm = None,
           dualflag=0):
    if nrm is None:
        nrm = numpy.max(numpy.abs(u))
    if dyn is None:
        dyn = 40

    ip = numpy.where(u > 0)
    im = numpy.where(u < 0)

    if im[0].size != 0:
        negflag = 1
        dualflag = 1

    udb = u / nrm

    if dualflag != 0:
        udb[ip] = 20.0 * numpy.log10(udb[ip])
        udb[im] = -20.0 * numpy.log10(-udb[im])
        inull = numpy.where(numpy.abs(udb) > dyn)

        udb[ip] = udb[ip] + dyn
        udb[im] = udb[im] - dyn

        udb[inull] = 0.0
    else:
        udb = 20.0 * numpy.log10(udb)
        inull = numpy.where(numpy.abs(udb) > dyn)
        udb[inull] = -dyn

    return udb


