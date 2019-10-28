from initpropcontrol import initpropcontrol
from visualization.makedb import makedb

import numpy
import matplotlib.pyplot as plt


def plot_beamprofile(prof,
                     propcontrol = None,
                     dyn = 40,
                     harm = 0,
                     nrm = None,
                     chan = None,
                     fh = None,
                     cmap = 'jet'):
    if propcontrol is None:
        propcontrol = initpropcontrol()

    if chan is None:
        chan = propcontrol.cchannel

    ny, nx, nz, nh = prof.shape
    if harm == 0:
        harm = numpy.arange(nh)
    nh = numpy.max(harm.shape)

    x = numpy.arange(-numpy.floor(nx/2), numpy.ceil(nx/2)) * propcontrol.dx * 1e3
    y = numpy.arange(-numpy.floor(ny/2), numpy.ceil(ny/2)) * propcontrol.dy * 1e3
    z = numpy.arange(nz) * propcontrol.stepsize * 1e3
    if propcontrol.annflag != 0 and propcontrol.ndims == 2:
        x = numpy.arange(0, nx) * propcontrol.dx * 1e3

    k = 0
    for ii in range(nh):
        sz = prof[..., harm[ii]].shape
        tmp = numpy.squeeze(prof[..., harm[ii]])
        tmp = tmp + numpy.finfo(float).eps
        tmpndim = tmp.ndim

        if ii == 0:
            if tmpndim <= 2:
                fig, axs = plt.subplots(nh, 1)
            elif tmpndim == 3:
                fig, axs = plt.subplots(nh, 2)
            htstr = 'total field'
        else:
            htstr = '{}. harmonic'.format(ii)

        if tmpndim == 3:
            data = makedb(numpy.squeeze(tmp[int(chan[1]), :, :]), dyn, nrm)
            axs[ii, k].imshow(data,
                             cmap=cmap,
                             aspect='auto',
                             extent=([numpy.min(z), numpy.max(z), numpy.max(x), numpy.min(x)]))
            axs[ii, k].set_xlabel('Depth [mm]')
            axs[ii, k].set_ylabel('Azimuth [mm]')
            axs[ii, k].set_title('Az. {}'.format(htstr))

            data = makedb(numpy.squeeze(tmp[:, int(chan[0]), :]), dyn, nrm)
            axs[ii, k + 1].imshow(data,
                             cmap=cmap,
                             aspect='auto',
                             extent=([numpy.min(z), numpy.max(z), numpy.max(y), numpy.min(y)]))
            axs[ii, k + 1].set_xlabel('Depth [mm]')
            axs[ii, k + 1].set_ylabel('Elevation [mm]')
            axs[ii, k + 1].set_title('El. {}'.format(htstr))
        elif tmpndim == 2 and numpy.prod(tmp.shape) != numpy.max(tmp.shape):
            if sz[0] == 1:
                tstr = 'Az.'
                xax = z
                xstr = 'Depth [mm]'
                yax = x
                ystr = 'Azimuth [mm]'
            elif sz[1] == 1:
                tstr = 'El.'
                xax = z
                xstr = 'Depth [mm]'
                yax = y
                ystr = 'Elevation [mm]'
            else:
                tstr = 'Crossct.'
                xax = x
                ystr = 'Elevation [mm]'
                yax = y
                xstr = 'Azimuth [mm]'
            if nh > 1:
                axs[k].imshow(makedb(tmp, dyn, nrm),
                              cmap=cmap,
                              aspect='auto',
                              extent=([0, numpy.max(xax), numpy.min(yax), numpy.max(yax)]))
                axs[k].set_xlabel(xstr)
                axs[k].set_ylabel(ystr)
                axs[k].set_title('{} {}'.format(tstr, htstr))
                k = k + 1
            else:
                raise NotImplementedError
        else:
            if dyn > 0:
                if nrm is None:
                    nrm = numpy.max(tmp)
                tmp = 20 * numpy.log10(tmp / nrm)
                offset = numpy.max(tmp)
                ystr = '[dB]'
            else:
                ystr = ''
            if sz[0] == 1:
                tstr = 'Az.'
                xax = x
                xstr = 'Azimuth [mm]'
            elif sz[1] == 1:
                tstr = 'El.'
                xax = y
                xstr = 'Elevation [mm]'
            if nh > 1:
                axs[k].plot(xax, tmp)
                axs[k].set_xlim(xax[0], xax[-1])
                axs[k].set_ylim(-dyn + offset, offset)
                axs[k].set_xlabel(xstr)
                axs[k].set_ylabel(ystr)
                axs[k].set_title('{} {}'.format(tstr, htstr))
                k = k + 1
            else:
                raise NotImplementedError
    plt.show()