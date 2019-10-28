from transducer.pulsegenerator import pulsegenerator
from initpropcontrol import initpropcontrol
from misc.find_steps import find_steps
from misc.get_window import get_window
from misc.estimate_eta import estimate_eta
from propagation.get_wavenumbers import get_wavenumbers
from postprocessing.export_beamprofile import export_beamprofile
from simscript.bodywall import bodywall
from propagation.propagate import propagate
from propcontrol import PropControlDataDecoder

import numpy
import scipy.sparse
import json
import codecs
import time


def beamsim(propcontrol = None,
            u_z = None,
            screen = numpy.array([]),
            w = None,
            phantom = None):

    if propcontrol is None:
        propcontrol = initpropcontrol()

    if u_z is None:
        u_z = pulsegenerator(propcontrol, 'transducer')

    # calculate number of propagation steps beyond the body wall
    currentpos = propcontrol.currentpos
    endpoint = propcontrol.endpoint
    stepsize = propcontrol.stepsize
    storepos = propcontrol.storepos

    if propcontrol.abflag != 0:
        if propcontrol.currentpos >= propcontrol.d:
            nsteps, step, stepidx = find_steps(currentpos, endpoint, stepsize, storepos)
        else:
            nsteps, step, stepidx = find_steps(propcontrol.d, endpoint, stepsize, storepos)
    else:
        nsteps, step, stepidx = find_steps(currentpos, endpoint, stepsize, storepos)

    # adjust equidistant stepsize flag
    dstepidx = numpy.concatenate((numpy.array([0], dtype=int), numpy.diff(stepidx)))
    nrecalc = len(numpy.where(dstepidx)[0])

    if propcontrol.diffrflag == 1 or propcontrol.diffrflag == 3:
        if nrecalc / nsteps < 0.5 / propcontrol.diffrflag:
            dorecalc = 1
            if stepidx[0] == 0:
                propcontrol.equidistflag = 1
        else:
            propcontrol.equidistflag = 0
            dorecalc = 0
    else:
        dorecalc = 0

    # sets sizes
    nx = propcontrol.nx
    ny = propcontrol.ny
    nt = propcontrol.nt
    nonlinflag = propcontrol.nonlinflag
    annflag = propcontrol.annflag
    historyflag = propcontrol.historyflag
    ndim = propcontrol.ndims
    dx = propcontrol.dx
    dy = propcontrol.dy

    # initializing variables
    if screen.size != 0:
        raise NotImplementedError
    fn = propcontrol.simname

    global Kz
    Kz = get_wavenumbers(propcontrol)

    # calculate spatial window
    if w is None:
        w = propcontrol.nwindow
    if isinstance(w, int) and w > 0:
        w = get_window((nx, ny), (dx, dy), w * stepsize, 2 * stepsize, annflag)

    t = numpy.zeros(nsteps + 1)
    tlap = 0

    # reporting simulation type
    if nonlinflag == 0:
        nstr = 'linear'
    else:
        nstr = 'non-linear'
    if ndim == 2:
        dstr = '{} x {}'.format(nx, nt)
    else:
        dstr = '{} x {} x {}'.format(nx, ny, nt)
    print('starting {} simulation of size {}'.format(nstr, dstr))

    # calculating beam profiles
    if historyflag != 0:
        rmspro = numpy.zeros((ny, nx, nsteps, propcontrol.harmonic+1))
        maxpro = numpy.zeros((ny, nx, nsteps, propcontrol.harmonic+1))
        axplse = numpy.zeros((nt, nsteps))
        zpos = numpy.zeros(nsteps)
    else:
        rmspro = numpy.array([])
        maxpro = numpy.array([])
        axplse = numpy.array([])
        zpos = numpy.array([])
    stepnr = 0

    rmspro, maxpro, axplse, zpos = export_beamprofile(u_z, propcontrol, rmspro, maxpro, axplse, zpos, stepnr)

    # Propagating through body wall
    if propcontrol.abflag != 0 and currentpos < propcontrol.d:
        print('Entering body wall')
        u_z = propcontrol, rmpro, mxpro, axpls, zps = bodywall(u_z, 1, propcontrol, Kz, w, phantom)
        print('Done with body wall')

        if historyflag != 0:
            pnx, pny, pns, pnh = rmpro.shape
            stepnr = pns
            raise NotImplementedError

        del rmpro
        del mxpro
        del axpls

    # Make window into sparse matrix
    if w[0] != -1:
        nw = numpy.max(w.shape)
        w = scipy.sparse.spdiags(w, 0, nw, nw)

    # Propagating the rest of the distance
    for ii in range(nsteps - 1):
        stepnr = stepnr + 1
        start_time = time.time()

        # recalculate wave number operator
        if dorecalc != 0:
            if dstepidx[ii] != 0:
                propcontrol.equidistflag = stepidx[ii]
                Kz = get_wavenumbers(propcontrol)

        # Propagation
        propcontrol.stepsize = step[ii]
        u_z, propcontrol = propagate(u_z, 1, propcontrol, Kz)

        # windowing of solution
        if w.data[0, 0] != -1:
            if ndim == 3:
                u_z = u_z.reshape((nt, nx * ny))
            u_z = u_z * w
            if ndim == 3:
                u_z = u_z.reshape((nt, ny, nx))

        # calculate beam profiles
        rmspro, maxpro, axplse, zpos = export_beamprofile(u_z, propcontrol, rmspro, maxpro, axplse, zpos, stepnr)

        toc = time.time() - start_time
        t[ii + 1] = t[ii] + toc
        tlap = estimate_eta(t, nsteps, ii, tlap)
    print('Simulation finished in {:.2f} min using an average of {} sec per step.'
          .format(t[-2] / 60.0, numpy.mean(numpy.diff(t[:-2]))))

    # saving the last profiles
    if historyflag == 2:
        data = {'rmspro': rmspro,
                'maxpro': maxpro,
                'axplse': axplse,
                'zpos': zpos,
                'control': propcontrol}
        json.dump(data, codecs.open(fn, 'w', encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4,
                  cls=PropControlDataDecoder)

    return u_z, propcontrol, rmspro, maxpro, axplse, zpos