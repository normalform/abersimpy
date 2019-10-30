import time

import numpy


def estimate_eta(t,
                 nsteps,
                 step,
                 tlap,
                 bwflag=0):
    if bwflag != 0:
        fstr = 'to end of body_wall'
    else:
        fstr = 'to final endpoint'
    step = step + 1
    if (numpy.mod(step, numpy.ceil(nsteps / 10)) == 0 or t[step] - tlap > 30) and step < nsteps:
        pcomp = step / nsteps
        prestime = time.localtime()
        tlap = t[step]
        ttot = t[step] / pcomp
        eta = t[step] - ttot
        eth = prestime.tm_hour + numpy.floor(-eta / 3600.0)
        eta = eta + numpy.floor(-eta / 3600.0) * 3600.0
        etm = prestime.tm_min + numpy.floor(-eta / 60.0)

        if etm >= 60:
            eeth = eth + 1
            etm = numpy.mod(etm, 60)
        if etm < 10:
            mstr = '{:1d}'.format(int(etm))
        else:
            mstr = '{:2d}'.format(int(etm))
        if eth >= 24:
            dstr = '+{:2d}'.format(int(numpy.floor(eth/24)))
            eth = numpy.mod(eth, 24)
        else:
            dstr = ''
        if eth < 10:
            hstr = '{:1d}'.format(int(eth))
        else:
            hstr = '{:2d}'.format(int(eth))
        tstr = '{}:{} {}'.format(hstr, mstr, dstr)

        print('Simulation {} is {:2.2f} percent complete (st. {}/{}), ETA {}'.format(fstr, 100 * pcomp, step, nsteps, tstr))
    else:
        tlap = tlap

    return tlap



