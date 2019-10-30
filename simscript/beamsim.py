import time

import numpy
import scipy.sparse

from consts import NoHistory, ProfileHistory
from misc.estimate_eta import estimate_eta
from misc.find_steps import find_steps
from misc.get_window import get_window
from postprocessing.export_beamprofile import export_beamprofile
from prop_control import ExactDiffraction, PseudoDifferential
from propagation.get_wavenumbers import get_wavenumbers
from propagation.propagate import propagate
from simscript.body_wall import body_wall
from transducer.pulsegenerator import pulsegenerator


def beamsim(prop_control=None,
            u_z = None,
            screen = numpy.array([]),
            w = None,
            phantom = None):
    if prop_control is None:
        prop_control = init_prop_control()

    if u_z is None:
        u_z = pulsegenerator(prop_control, 'transducer')

    # calculate number of propagation steps beyond the body wall
    currentpos = prop_control.currentpos
    endpoint = prop_control.endpoint
    stepsize = prop_control.stepsize
    storepos = prop_control.storepos

    if prop_control.config.heterogeneous_medium:
        if prop_control.currentpos >= prop_control.d:
            nsteps, step, stepidx = find_steps(currentpos, endpoint, stepsize, storepos)
        else:
            nsteps, step, stepidx = find_steps(prop_control.d, endpoint, stepsize, storepos)
    else:
        nsteps, step, stepidx = find_steps(currentpos, endpoint, stepsize, storepos)

    # adjust equidistant stepsize flag
    dstepidx = numpy.concatenate((numpy.array([0], dtype=int), numpy.diff(stepidx)))
    nrecalc = len(numpy.where(dstepidx)[0])

    if prop_control.config.diffraction_type == ExactDiffraction or \
            prop_control.config.diffraction_type == PseudoDifferential:
        if prop_control.config.diffraction_type == ExactDiffraction:
            diffraction_factor = 1
        else:
            diffraction_factor = 3
        if nrecalc / nsteps < 0.5 / diffraction_factor:
            dorecalc = 1
            if stepidx[0] == 0:
                # TODO do not change the member directly
                prop_control.config.equidistant_steps = True
        else:
            # TODO do not change the member directly
            prop_control.config.equidistant_steps = False
            dorecalc = 0
    else:
        dorecalc = 0

    # sets sizes
    nx = prop_control.nx
    ny = prop_control.ny
    nt = prop_control.nt
    non_linearity = prop_control.config.non_linearity
    annular_transducer = prop_control.config.annular_transducer
    history = prop_control.config.history
    num_dimensions = prop_control.num_dimensions
    dx = prop_control.dx
    dy = prop_control.dy

    # initializing variables
    if screen.size != 0:
        raise NotImplementedError
    fn = prop_control.simulation_name

    global Kz
    Kz = get_wavenumbers(prop_control)

    # calculate spatial window
    if w is None:
        w = prop_control.nwindow
    if isinstance(w, int) and w > 0:
        w = get_window((nx, ny), (dx, dy), w * stepsize, 2 * stepsize, annular_transducer)

    t = numpy.zeros(nsteps + 1)
    tlap = 0

    # reporting simulation type
    if non_linearity:
        nstr = 'non-linear'
    else:
        nstr = 'linear'
    if num_dimensions == 2:
        dstr = '{} x {}'.format(nx, nt)
    else:
        dstr = '{} x {} x {}'.format(nx, ny, nt)
    print('starting {} simulation of size {}'.format(nstr, dstr))

    # calculating beam profiles
    if history != NoHistory:
        rmspro = numpy.zeros((ny, nx, nsteps, prop_control.harmonic + 1))
        maxpro = numpy.zeros((ny, nx, nsteps, prop_control.harmonic + 1))
        axplse = numpy.zeros((nt, nsteps))
        zpos = numpy.zeros(nsteps)
    else:
        rmspro = numpy.array([])
        maxpro = numpy.array([])
        axplse = numpy.array([])
        zpos = numpy.array([])
    stepnr = 0

    rmspro, maxpro, axplse, zpos = export_beamprofile(u_z, prop_control, rmspro, maxpro, axplse, zpos, stepnr)

    # Propagating through body wall
    if prop_control.config.heterogeneous_medium != 0 and currentpos < prop_control.d:
        print('Entering body wall')
        u_z = prop_control, rmpro, mxpro, axpls, zps = body_wall(u_z, 1, prop_control, Kz, w, phantom)
        print('Done with body wall')

        if history != NoHistory:
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
                # TODO do not change directly
                if stepidx[ii] == 0:
                    prop_control.config.equidistant_steps = False
                else:
                    prop_control.config.equidistant_steps = True
                Kz = get_wavenumbers(prop_control)

        # Propagation
        prop_control.stepsize = step[ii]
        u_z, prop_control = propagate(u_z, 1, prop_control, Kz)

        # windowing of solution
        if w.data[0, 0] != -1:
            if num_dimensions == 3:
                u_z = u_z.reshape((nt, nx * ny))
            u_z = u_z * w
            if num_dimensions == 3:
                u_z = u_z.reshape((nt, ny, nx))

        # calculate beam profiles
        rmspro, maxpro, axplse, zpos = export_beamprofile(u_z, prop_control, rmspro, maxpro, axplse, zpos, stepnr)

        toc = time.time() - start_time
        t[ii + 1] = t[ii] + toc
        tlap = estimate_eta(t, nsteps, ii, tlap)
    print('Simulation finished in {:.2f} min using an average of {} sec per step.'
          .format(t[-2] / 60.0, numpy.mean(numpy.diff(t[:-2]))))

    # saving the last profiles
    if history == ProfileHistory:
        # TODO Current (removed) json Serialization is not working well.
        print('[DUMMY] Saving the last profiles')

    return u_z, prop_control, rmspro, maxpro, axplse, zpos
