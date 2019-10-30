import time

import numpy
import scipy.sparse

from consts import NoHistory, ProfileHistory
from misc.estimate_eta import estimate_eta
from misc.find_steps import find_steps
from misc.get_window import get_window
from postprocessing.export_beamprofile import export_beamprofile
from prop_control import PropControl, ExactDiffraction, PseudoDifferential
from propagation.get_wavenumbers import get_wavenumbers
from propagation.propagate import propagate
from simscript.body_wall import body_wall
from transducer.pulsegenerator import pulsegenerator


def beam_simulation(prop_control=None,
                    u_z=None,
                    screen=numpy.array([]),
                    w=None,
                    phantom=None):
    if prop_control is None:
        prop_control = PropControl.init_prop_control()

    if u_z is None:
        u_z = pulsegenerator(prop_control, 'transducer')

    # calculate number of propagation steps beyond the body wall
    current_pos = prop_control.currentpos
    end_point = prop_control.endpoint
    step_size = prop_control.step_size
    store_pos = prop_control.storepos

    if prop_control.config.heterogeneous_medium:
        if prop_control.currentpos >= prop_control.d:
            num_steps, step, step_idx = find_steps(current_pos, end_point, step_size, store_pos)
        else:
            num_steps, step, step_idx = find_steps(prop_control.d, end_point, step_size, store_pos)
    else:
        num_steps, step, step_idx = find_steps(current_pos, end_point, step_size, store_pos)

    # adjust equidistant step size flag
    diff_step_idx = numpy.concatenate((numpy.array([0], dtype=int), numpy.diff(step_idx)))
    num_recalculate = len(numpy.where(diff_step_idx)[0])

    if prop_control.config.diffraction_type == ExactDiffraction or \
            prop_control.config.diffraction_type == PseudoDifferential:
        if prop_control.config.diffraction_type == ExactDiffraction:
            diffraction_factor = 1
        else:
            diffraction_factor = 3
        if num_recalculate / num_steps < 0.5 / diffraction_factor:
            recalculate = True
            if step_idx[0] == 0:
                # TODO do not change the member directly
                prop_control.config.equidistant_steps = True
        else:
            # TODO do not change the member directly
            prop_control.config.equidistant_steps = False
            recalculate = False
    else:
        recalculate = False

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
    file_name = prop_control.simulation_name

    global Kz
    Kz = get_wavenumbers(prop_control)

    # calculate spatial window
    if w is None:
        w = prop_control.nwindow
    if isinstance(w, int) and w > 0:
        w = get_window((nx, ny), (dx, dy), w * step_size, 2 * step_size, annular_transducer)

    t = numpy.zeros(num_steps + 1)
    t_lap = 0

    # reporting simulation type
    if non_linearity:
        non_linearity_str = 'non-linear'
    else:
        non_linearity_str = 'linear'
    if num_dimensions == 2:
        dimensions_str = '{} x {}'.format(nx, nt)
    else:
        dimensions_str = '{} x {} x {}'.format(nx, ny, nt)
    print('starting {} simulation of size {}'.format(non_linearity_str, dimensions_str))

    # calculating beam profiles
    if history != NoHistory:
        rms_pro = numpy.zeros((ny, nx, num_steps, prop_control.harmonic + 1))
        max_pro = numpy.zeros((ny, nx, num_steps, prop_control.harmonic + 1))
        ax_pulse = numpy.zeros((nt, num_steps))
        z_pos = numpy.zeros(num_steps)
    else:
        rms_pro = numpy.array([])
        max_pro = numpy.array([])
        ax_pulse = numpy.array([])
        z_pos = numpy.array([])
    step_nr = 0

    rms_pro, max_pro, ax_pulse, z_pos = export_beamprofile(u_z, prop_control, rms_pro, max_pro, ax_pulse, z_pos,
                                                           step_nr)

    # Propagating through body wall
    if prop_control.config.heterogeneous_medium != 0 and current_pos < prop_control.d:
        print('Entering body wall')
        u_z, prop_control, _rms_pro, _max_pro, _ax_pulse, _z_pos = body_wall(u_z, 1, prop_control, Kz, w, phantom)
        print('Done with body wall')

        if history != NoHistory:
            pnx, pny, pns, pnh = _rms_pro.shape
            step_nr = pns
            raise NotImplementedError

    # Make window into sparse matrix
    if w[0] != -1:
        nw = numpy.max(w.shape)
        w = scipy.sparse.spdiags(w, 0, nw, nw)

    # Propagating the rest of the distance
    for ii in range(num_steps - 1):
        step_nr = step_nr + 1
        start_time = time.time()

        # recalculate wave number operator
        if recalculate:
            if diff_step_idx[ii] != 0:
                # TODO do not change directly
                if step_idx[ii] == 0:
                    prop_control.config.equidistant_steps = False
                else:
                    prop_control.config.equidistant_steps = True
                Kz = get_wavenumbers(prop_control)

        # Propagation
        prop_control.step_size = step[ii]
        u_z, prop_control = propagate(u_z, 1, prop_control, Kz)

        # windowing of solution
        if w.data[0, 0] != -1:
            if num_dimensions == 3:
                u_z = u_z.reshape((nt, nx * ny))
            u_z = u_z * w
            if num_dimensions == 3:
                u_z = u_z.reshape((nt, ny, nx))

        # calculate beam profiles
        rms_pro, max_pro, ax_pulse, z_pos = export_beamprofile(u_z, prop_control, rms_pro, max_pro, ax_pulse, z_pos,
                                                               step_nr)

        toc = time.time() - start_time
        t[ii + 1] = t[ii] + toc
        t_lap = estimate_eta(t, num_steps, ii, t_lap)
    print('Simulation finished in {:.2f} min using an average of {} sec per step.'
          .format(t[-2] / 60.0, numpy.mean(numpy.diff(t[:-2]))))

    # saving the last profiles
    if history == ProfileHistory:
        # TODO Current (removed) json Serialization is not working well.
        print('[DUMMY] Saving the last profiles')

    return u_z, prop_control, rms_pro, max_pro, ax_pulse, z_pos
