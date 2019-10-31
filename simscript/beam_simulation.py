import time

import numpy
import scipy.sparse

from controls.consts import NoHistory, ProfileHistory
from controls.main_control import ExactDiffraction, PseudoDifferential
from misc.estimate_eta import estimate_eta
from misc.find_steps import find_steps
from misc.get_window import get_window
from postprocessing.export_beamprofile import export_beamprofile
from propagation.get_wavenumbers import get_wavenumbers
from propagation.propagate import propagate
from simscript.body_wall import body_wall
from transducer.pulsegenerator import pulsegenerator


def beam_simulation(main_control,
                    u_z=None,
                    screen=numpy.array([]),
                    w=None,
                    phantom=None):
    if u_z is None:
        u_z = pulsegenerator(main_control, 'transducer')

    # calculate number of propagation steps beyond the body wall
    current_pos = main_control.current_position
    end_point = main_control.endpoint
    step_size = main_control.step_size
    store_pos = main_control.store_position

    if main_control.config.heterogeneous_medium:
        if main_control.current_position >= main_control.thickness:
            num_steps, step, step_idx = find_steps(current_pos, end_point, step_size, store_pos)
        else:
            num_steps, step, step_idx = find_steps(main_control.thickness, end_point, step_size, store_pos)
    else:
        num_steps, step, step_idx = find_steps(current_pos, end_point, step_size, store_pos)

    # adjust equidistant step size flag
    diff_step_idx = numpy.concatenate((numpy.array([0], dtype=int), numpy.diff(step_idx)))
    num_recalculate = len(numpy.where(diff_step_idx)[0])

    if main_control.config.diffraction_type == ExactDiffraction or \
            main_control.config.diffraction_type == PseudoDifferential:
        if main_control.config.diffraction_type == ExactDiffraction:
            diffraction_factor = 1
        else:
            diffraction_factor = 3
        if num_recalculate / num_steps < 0.5 / diffraction_factor:
            recalculate = True
            if step_idx[0] == 0:
                # TODO do not change the member directly
                main_control.config.equidistant_steps = True
        else:
            # TODO do not change the member directly
            main_control.config.equidistant_steps = False
            recalculate = False
    else:
        recalculate = False

    # sets sizes
    num_points_x = main_control.num_points_x
    num_points_y = main_control.num_points_y
    num_points_t = main_control.num_points_t
    non_linearity = main_control.config.non_linearity
    annular_transducer = main_control.config.annular_transducer
    history = main_control.config.history
    num_dimensions = main_control.num_dimensions
    resolution_x = main_control.resolution_x
    resolution_y = main_control.resolution_y

    # initializing variables
    if screen.size != 0:
        raise NotImplementedError
    file_name = main_control.simulation_name

    global Kz
    Kz = get_wavenumbers(main_control)

    # calculate spatial window
    if w is None:
        w = main_control.num_windows
    if isinstance(w, int) and w > 0:
        w = get_window((num_points_x, num_points_y), (resolution_x, resolution_y), w * step_size, 2 * step_size,
                       annular_transducer)

    t = numpy.zeros(num_steps + 1)
    t_lap = 0

    # reporting simulation type
    if non_linearity:
        non_linearity_str = 'non-linear'
    else:
        non_linearity_str = 'linear'
    if num_dimensions == 2:
        dimensions_str = '{} x {}'.format(num_points_x, num_points_t)
    else:
        dimensions_str = '{} x {} x {}'.format(num_points_x, num_points_y, num_points_t)
    print('starting {} simulation of size {}'.format(non_linearity_str, dimensions_str))

    # calculating beam profiles
    if history != NoHistory:
        rms_pro = numpy.zeros((num_points_y, num_points_x, num_steps, main_control.harmonic + 1))
        max_pro = numpy.zeros((num_points_y, num_points_x, num_steps, main_control.harmonic + 1))
        ax_pulse = numpy.zeros((num_points_t, num_steps))
        z_pos = numpy.zeros(num_steps)
    else:
        rms_pro = numpy.array([])
        max_pro = numpy.array([])
        ax_pulse = numpy.array([])
        z_pos = numpy.array([])
    step_nr = 0

    rms_pro, max_pro, ax_pulse, z_pos = export_beamprofile(u_z, main_control, rms_pro, max_pro, ax_pulse, z_pos,
                                                           step_nr)

    # Propagating through body wall
    if main_control.config.heterogeneous_medium != 0 and current_pos < main_control.thickness:
        print('Entering body wall')
        u_z, main_control, _rms_pro, _max_pro, _ax_pulse, _z_pos = body_wall(u_z, 1, main_control, Kz, w, phantom)
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
                    main_control.config.equidistant_steps = False
                else:
                    main_control.config.equidistant_steps = True
                Kz = get_wavenumbers(main_control)

        # Propagation
        main_control.step_size = step[ii]
        u_z, main_control = propagate(u_z, 1, main_control, Kz)

        # windowing of solution
        if w.data[0, 0] != -1:
            if num_dimensions == 3:
                u_z = u_z.reshape((num_points_t, num_points_x * num_points_y))
            u_z = u_z * w
            if num_dimensions == 3:
                u_z = u_z.reshape((num_points_t, num_points_y, num_points_x))

        # calculate beam profiles
        rms_pro, max_pro, ax_pulse, z_pos = export_beamprofile(u_z, main_control, rms_pro, max_pro, ax_pulse, z_pos,
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

    return u_z, main_control, rms_pro, max_pro, ax_pulse, z_pos
