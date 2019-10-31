import matplotlib.pyplot as plt
import numpy

from visualization.makedb import makedb


def plot_beamprofile(prof,
                     main_control,
                     dyn=40,
                     harm=0,
                     nrm=None,
                     chan=None,
                     fh=None,
                     cmap='jet'):
    if chan is None:
        chan = main_control.transducer.center_channel

    num_points_y, num_points_x, nz, nh = prof.shape
    if harm == 0:
        harm = numpy.arange(nh)
    nh = numpy.max(harm.shape)

    x = numpy.arange(-numpy.floor(num_points_x / 2),
                     numpy.ceil(num_points_x / 2)) * main_control.signal.resolution_x * 1e3
    y = numpy.arange(-numpy.floor(num_points_y / 2),
                     numpy.ceil(num_points_y / 2)) * main_control.signal.resolution_y * 1e3
    z = numpy.arange(nz) * main_control.simulation.step_size * 1e3
    if main_control.config.annular_transducer and main_control.num_dimensions == 2:
        x = numpy.arange(0, num_points_x) * main_control.signal.resolution_x * 1e3

    k = 0
    for ii in range(nh):
        sz = prof[..., harm[ii]].shape
        tmp = numpy.squeeze(prof[..., harm[ii]])
        tmp = tmp + numpy.finfo(float).eps
        tmp_num_dimensions = tmp.ndim

        if ii == 0:
            if tmp_num_dimensions <= 2:
                fig, axs = plt.subplots(nh, 1)
            elif tmp_num_dimensions == 3:
                fig, axs = plt.subplots(nh, 2)
            htstr = 'total field'
        else:
            htstr = '{}. harmonic'.format(ii)

        if tmp_num_dimensions == 3:
            fig.canvas.set_window_title('Beam profile 3 Dim')
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
        elif tmp_num_dimensions == 2 and numpy.prod(tmp.shape) != numpy.max(tmp.shape):
            fig.canvas.set_window_title('Beam profile 2 Dim')
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
            fig.canvas.set_window_title('Beam profile 1 Dim')
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
