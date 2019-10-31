import numpy

from consts import NoHistory, PositionHistory, ProfileHistory, FullHistory, PlaneHistory, PlaneByChannelHistory
from filter.bandpass import bandpass
from misc.get_strpos import get_strpos
from postprocessing.get_max import get_max
from postprocessing.get_rms import get_rms


def export_beamprofile(u_z,
                       prop_control,
                       rmpro=None,
                       mxpro=None,
                       axpls=None,
                       zps=None,
                       step=None):
    # setting variables
    filename = prop_control.simulation_name
    rmspro = None
    maxpro = None
    axplse = None
    zpos = None

    history = prop_control.config.history
    pos = prop_control.current_position
    fn = '{}{}.json'.format(filename, get_strpos(pos * 1e3))

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

    harmonic = prop_control.harmonic
    store_position = prop_control.store_position
    cc = prop_control.center_channel.astype(int)

    transmit_frequency = prop_control.transmit_frequency
    resolution_t = prop_control.resolution_t
    filterc = prop_control.filter

    # initializing profiles
    rmspro = rmpro
    maxpro = mxpro
    axplse = axpls
    zpos = zps

    # finding dimensions
    num_dimensions = prop_control.num_dimensions
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
            # TODO Current (removed) json Serialization is not working well.
            print('[DUMMY] save pulse to {}'.format(fn))

    if history == PositionHistory:
        return rmspro, maxpro, axplse, zpos
    elif history == PlaneByChannelHistory:
        raise NotImplementedError
    elif history == ProfileHistory:
        if num_dimensions == 2:
            axplse[:, num_periods] = u_z[:, cc[0, ...]]
        else:
            axplse[:, num_periods] = u_z[:, cc[1], cc[0, ...]]
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
            tmp, _ = bandpass(tmp, idx * transmit_frequency, resolution_t, idx * transmit_frequency * filterc[ii], 4)
            rms = get_rms(tmp, 1)
            max = get_max(tmp, 1)
            rmspro[..., num_periods, ii + 1] = rms.reshape((num_points_y, num_points_x))
            maxpro[..., num_periods, ii + 1] = max.reshape((num_points_y, num_points_x))
    elif history == PlaneHistory:
        raise NotImplementedError

    return rmspro, maxpro, axplse, zpos
