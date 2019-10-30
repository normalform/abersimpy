import numpy

from consts import NoHistory, PositionHistory, ProfileHistory, FullHistory, PlaneHistory, PlaneByChannelHistory
from filter.bandpass import bandpass
from misc.get_strpos import get_strpos
from postprocessing.get_max import get_max
from postprocessing.get_rms import get_rms


def export_beamprofile(u_z,
                       prop_control,
                       rmpro = None,
                       mxpro = None,
                       axpls = None,
                       zps = None,
                       step = None):
    # setting variables
    filename = prop_control.simulation_name
    rmspro = None
    maxpro = None
    axplse = None
    zpos = None

    history = prop_control.config.history
    pos = prop_control.currentpos
    fn = '{}{}.json'.format(filename, get_strpos(pos * 1e3))

    # stores full field or exits
    if history == NoHistory:
        # No history --> Exit
        raise  NotImplementedError
        return rmspro, maxpro, axplse, zpos
    elif history == FullHistory:
        # Saving pulse for each step, then exit
        print('save pulse to {}'.format(fn))
        raise NotImplementedError
        return rmspro, maxpro, axplse, zpos

    harmonic = prop_control.harmonic
    storepos = prop_control.storepos
    cc = prop_control.cchannel.astype(int)

    fc = prop_control.fc
    dt = prop_control.dt
    filterc = prop_control.filter

    # initializing profiles
    rmspro = rmpro
    maxpro = mxpro
    axplse = axpls
    zpos = zps

    # finding dimensions
    num_dimensions = prop_control.num_dimensions
    if num_dimensions == 2:
        nt, ny = u_z.shape
        nx = ny
        ny = 1
    else:
        nt, ny, nx = u_z.shape
    if step is None:
        np = 0
        zpos = 0
    else:
        np = step

    # saving pulse for steps specified in storepos
    if storepos.size != 0:
        if numpy.min(numpy.abs(storepos - pos)) < 1e-12:
            # TODO Current (removed) json Serialization is not working well.
            print('[DUMMY] save pulse to {}'.format(fn))

    if history == PositionHistory:
        return rmspro, maxpro, axplse, zpos
    elif history == PlaneByChannelHistory:
        raise NotImplementedError
    elif history == ProfileHistory:
        if num_dimensions == 2:
            axplse[:, np] = u_z[:, cc[0, ...]]
        else:
            axplse[:, np] = u_z[:, cc[1], cc[0, ...]]
        if np == 0:
            rmspro = numpy.zeros_like(rmspro)
            maxpro = numpy.zeros_like(maxpro)

        tmp = u_z.reshape((nt, nx * ny))
        rms = get_rms(tmp, 1)
        max = get_max(tmp, 1)
        rmspro[..., np, 0] = rms.reshape((ny, nx))
        maxpro[..., np, 0] = max.reshape((ny, nx))
        zpos [np] = pos

        # filtering out harmonics
        for ii in range(harmonic):
            idx = ii + 1
            tmp = u_z.reshape((nt, nx * ny))
            tmp, _ = bandpass(tmp, idx * fc, dt, idx*fc*filterc[ii], 4)
            rms = get_rms(tmp, 1)
            max = get_max(tmp, 1)
            rmspro[..., np, ii + 1] = rms.reshape((ny, nx))
            maxpro[..., np, ii + 1] = max.reshape((ny, nx))
    elif history == PlaneHistory:
        raise NotImplementedError

    return rmspro, maxpro, axplse, zpos