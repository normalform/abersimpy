from consts import NoHistory, PositionHistory, ProfileHistory, FullHistory, PlaneHistory, PlaneByChannelHistory
from misc.get_strpos import get_strpos
from postprocessing.get_rms import get_rms
from postprocessing.get_max import get_max
from filter.bandpass import bandpass

import numpy


def export_beamprofile(u_z,
                       propcontrol,
                       rmpro = None,
                       mxpro = None,
                       axpls = None,
                       zps = None,
                       step = None):
    # setting variables
    filename = propcontrol.simulation_name
    rmspro = None
    maxpro = None
    axplse = None
    zpos = None

    history = propcontrol.config.history
    pos = propcontrol.currentpos
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

    harmonic = propcontrol.harmonic
    storepos = propcontrol.storepos
    cc = propcontrol.cchannel.astype(int)

    fc = propcontrol.fc
    dt = propcontrol.dt
    filterc = propcontrol.filter

    # initializing profiles
    rmspro = rmpro
    maxpro = mxpro
    axplse = axpls
    zpos = zps

    # finding dimensions
    num_dimensions = propcontrol.num_dimensions
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