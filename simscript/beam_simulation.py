import time

import numpy
import scipy.sparse

from controls.consts import NoHistory, ProfileHistory
from diffraction.diffraction import ExactDiffraction, PseudoDifferential
from misc.estimate_eta import estimate_eta
from misc.find_steps import find_steps
from misc.get_window import get_window
from postprocessing.export_beamprofile import export_beamprofile
from propagation.get_wavenumbers import get_wavenumbers
from propagation.propagate import propagate
from simscript.body_wall import body_wall


def beam_simulation(control,
                    u_z,
                    screen=numpy.array([]),
                    w=None,
                    phantom=None):
    # calculate number of propagation steps beyond the body wall
    current_pos = control.simulation.current_position
    end_point = control.simulation.endpoint
    step_size = control.simulation.step_size
    store_pos = control.simulation.store_position
    _equidistant_steps = control.equidistant_steps

    if control.heterogeneous_medium:
        if control.simulation.current_position >= control.material.thickness:
            num_steps, step, step_idx = find_steps(current_pos, end_point, step_size, store_pos)
        else:
            num_steps, step, step_idx = find_steps(control.material.thickness, end_point, step_size,
                                                   store_pos)
    else:
        num_steps, step, step_idx = find_steps(current_pos, end_point, step_size, store_pos)

    # adjust equidistant step size flag
    diff_step_idx = numpy.concatenate((numpy.array([0], dtype=int), numpy.diff(step_idx)))
    num_recalculate = len(numpy.where(diff_step_idx)[0])

    if control.diffraction_type == ExactDiffraction or \
            control.diffraction_type == PseudoDifferential:
        if control.diffraction_type == ExactDiffraction:
            diffraction_factor = 1
        else:
            diffraction_factor = 3
        if num_recalculate / num_steps < 0.5 / diffraction_factor:
            recalculate = True
            if step_idx[0] == 0:
                _equidistant_steps = True
        else:
            _equidistant_steps = False
            recalculate = False
    else:
        recalculate = False

    # sets sizes
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t
    non_linearity = control.non_linearity
    annular_transducer = control.annular_transducer
    history = control.history
    num_dimensions = control.num_dimensions
    resolution_x = control.signal.resolution_x
    resolution_y = control.signal.resolution_y

    # initializing variables
    if screen.size != 0:
        raise NotImplementedError
    file_name = control.simulation_name

    global Kz
    Kz = get_wavenumbers(control, _equidistant_steps)

    # calculate spatial window
    if w is None:
        w = control.simulation.num_windows
    if isinstance(w, int) and w > 0:
        w = get_window((num_points_x, num_points_y), (resolution_x, resolution_y), w * step_size,
                       2 * step_size,
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
        rms_pro = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        max_pro = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        ax_pulse = numpy.zeros((num_points_t, num_steps))
        z_pos = numpy.zeros(num_steps)
    else:
        rms_pro = numpy.array([])
        max_pro = numpy.array([])
        ax_pulse = numpy.array([])
        z_pos = numpy.array([])
    step_nr = 0

    rms_pro, max_pro, ax_pulse, z_pos = export_beamprofile(u_z, control, rms_pro, max_pro, ax_pulse,
                                                           z_pos,
                                                           step_nr)

    # Propagating through body wall
    if control.heterogeneous_medium != 0 and current_pos < control.material.thickness:
        print('Entering body wall')
        u_z, control, _rms_pro, _max_pro, _ax_pulse, _z_pos = body_wall(u_z,
                                                                        1,
                                                                        control,
                                                                        _equidistant_steps,
                                                                        Kz,
                                                                        w,
                                                                        phantom)
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
                if step_idx[ii] == 0:
                    _equidistant_steps = False
                else:
                    _equidistant_steps = True
                Kz = get_wavenumbers(control, _equidistant_steps)

        # Propagation
        control.simulation.step_size = step[ii]
        u_z = propagate(u_z,
                        direction=1,
                        control=control,
                        equidistant_steps=_equidistant_steps,
                        Kz=Kz)

        # windowing of solution
        if w.data[0, 0] != -1:
            if num_dimensions == 3:
                u_z = u_z.reshape((num_points_t, num_points_x * num_points_y))
            u_z = u_z * w
            if num_dimensions == 3:
                u_z = u_z.reshape((num_points_t, num_points_y, num_points_x))

        # calculate beam profiles
        rms_pro, max_pro, ax_pulse, z_pos = export_beamprofile(u_z, control, rms_pro, max_pro,
                                                               ax_pulse, z_pos,
                                                               step_nr)

        toc = time.time() - start_time
        t[ii + 1] = t[ii] + toc
        t_lap = estimate_eta(t, num_steps, ii, t_lap)
    print('Simulation finished in {:.2f} min using an average of {} sec per step.'
          .format(t[-2] / 60.0, numpy.mean(numpy.diff(t[:-2]))))

    # saving the last profiles
    if history == ProfileHistory:
        print(f'[DUMMY] Saving the last profiles to {file_name}.json')

    return u_z, rms_pro, max_pro, ax_pulse, z_pos
