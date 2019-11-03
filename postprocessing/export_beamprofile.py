import numpy

from controls.consts import NoHistory, PositionHistory, ProfileHistory, FullHistory, PlaneHistory, \
    PlaneByChannelHistory
from filter.bandpass import bandpass
from misc.get_string_position import get_string_position
from postprocessing.get_max import get_max
from postprocessing.get_rms import get_rms


def export_beamprofile(control,
                       u_z,
                       rmpro=None,
                       mxpro=None,
                       axpls=None,
                       zps=None,
                       step=None):
    # setting variables
    filename = control.simulation_name
    rmspro = None
    maxpro = None
    axplse = None
    zpos = None

    history = control.history
    pos = control.simulation.current_position
    fn = '{}{}.json'.format(filename, get_string_position(pos * 1e3))

    # stores full field or exits
    if history == NoHistory:
        # No history --> Exit
        raise NotImplementedError
        return rmspro, maxpro, axplse, zpos
    elif history == FullHistory:
        # Saving pulse for each step, then exit
        print('save pulse to {}'.format(fn))
        raise NotImplementedError
        return rmspro, maxpro, axplse, zpos

    harmonic = control.harmonic
    store_position = control.simulation.store_position
    center_channel = control.transducer.center_channel.astype(int)

    transmit_frequency = control.signal.transmit_frequency
    resolution_t = control.signal.resolution_t
    filterc = control.signal.filter

    # initializing profiles
    rmspro = rmpro
    maxpro = mxpro
    axplse = axpls
    zpos = zps

    # finding dimensions
    num_dimensions = control.num_dimensions
    if num_dimensions == 2:
        num_points_t, num_points_y = u_z.shape
        num_points_x = num_points_y
        num_points_y = 1
    else:
        num_points_t, num_points_y, num_points_x = u_z.shape
    if step is None:
        num_periods = 0
        zpos = 0
    else:
        num_periods = step

    # saving pulse for steps specified in store_position
    if store_position.size != 0:
        if numpy.min(numpy.abs(store_position - pos)) < 1e-12:
            print('[DUMMY] save pulse to {}'.format(fn))

    if history == PositionHistory:
        return rmspro, maxpro, axplse, zpos
    elif history == PlaneByChannelHistory:
        raise NotImplementedError
    elif history == ProfileHistory:
        if num_dimensions == 2:
            axplse[:, num_periods] = u_z[:, center_channel[0, ...]]
        else:
            axplse[:, num_periods] = u_z[:, center_channel[1], center_channel[0, ...]]
        if num_periods == 0:
            rmspro = numpy.zeros_like(rmspro)
            maxpro = numpy.zeros_like(maxpro)

        tmp = u_z.reshape((num_points_t, num_points_x * num_points_y))
        rms = get_rms(tmp, 1)
        max = get_max(tmp, 1)
        rmspro[..., num_periods, 0] = rms.reshape((num_points_y, num_points_x))
        maxpro[..., num_periods, 0] = max.reshape((num_points_y, num_points_x))
        zpos[num_periods] = pos

        # filtering out harmonics
        for ii in range(harmonic):
            idx = ii + 1
            tmp = u_z.reshape((num_points_t, num_points_x * num_points_y))
            tmp, _ = bandpass(tmp,
                              numpy.array([idx * transmit_frequency]),
                              resolution_t,
                              idx * transmit_frequency * filterc[ii],
                              4)
            rms = get_rms(tmp, 1)
            max = get_max(tmp, 1)
            rmspro[..., num_periods, ii + 1] = rms.reshape((num_points_y, num_points_x))
            maxpro[..., num_periods, ii + 1] = max.reshape((num_points_y, num_points_x))
    elif history == PlaneHistory:
        raise NotImplementedError

    return rmspro, maxpro, axplse, zpos
