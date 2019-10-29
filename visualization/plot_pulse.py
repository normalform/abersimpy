from initpropcontrol import initpropcontrol
from visualization.makedb import makedb
from filter.bandpass import bandpass

import numpy
from scipy.signal import hilbert
import matplotlib.pyplot as plt


def plot_pulse(u,
               propcontrol = None,
               axflag = 0,
               dyn = 40,
               nrm = None,
               fh = None,
               cmap = 'gray'):
    if propcontrol is None:
        propcontrol = initpropcontrol()

    ndim = u.ndim
    nt = u.shape[0]
    ny = u.shape[1]
    if ndim == 2:
        nx = ny
    else:
        nx = u.shape[2]

    cc = propcontrol.cchannel
    t = numpy.arange(-numpy.floor(nt / 2), numpy.ceil(nt / 2)) * propcontrol.dt * 1e6
    y = numpy.arange(-numpy.floor(ny / 2), numpy.ceil(ny / 2)) * propcontrol.dy * 1e3

    fig, axs = plt.subplots(2, 2)
    fig.canvas.set_window_title('Pulse')
    if axflag == 0:
        x = numpy.arange(-numpy.floor(nx / 2), numpy.ceil(nx / 2)) * propcontrol.dx * 1e3
        if propcontrol.annflag != 0 and propcontrol.ndims == 2:
            x = numpy.arange(nx) * propcontrol.dx * 1e3

        if ndim == 3:
            data = makedb(numpy.transpose(numpy.squeeze(u[:, int(cc[1]), :])), dyn, nrm)
            axs[0, 0].imshow(data,
                             cmap=cmap,
                             aspect='auto',
                             extent=([numpy.min(t), numpy.max(t), numpy.max(x), numpy.min(x)]))
            axs[0, 0].set_xlabel('Time [us]')
            axs[0, 0].set_ylabel('Azimuth [mm]')

            axs[0, 1].plot(numpy.squeeze(u[:, int(cc[1]), int(cc[0])]))
            axs[0, 1].set_xlabel('Time [us]')
            axs[0, 1].set_ylabel('Pressure [MPa]')

            data = makedb(numpy.transpose(numpy.squeeze(u[:, :, int(cc[0])])), dyn, nrm)
            axs[1, 0].imshow(data,
                             cmap=cmap,
                             aspect='auto',
                             extent=([numpy.min(t), numpy.max(t), numpy.max(x), numpy.min(x)]))
            axs[1, 0].set_xlabel('Time [us]')
            axs[1, 0].set_ylabel('Elevation [mm]')

            data = makedb(numpy.squeeze(numpy.sqrt(numpy.sum(u ** 2, axis=0))), dyn, nrm)
            axs[1, 1].imshow(data,
                             cmap=cmap,
                             aspect='auto',
                             extent=([numpy.min(x), numpy.max(x), numpy.min(y), numpy.max(y)]))
            axs[1, 1].set_xlabel('Azimuth [mm]')
            axs[1, 1].set_ylabel('Elevation [mm]')
        else:
            raise NotImplementedError
    else:
        x = numpy.arange(nx) * propcontrol.stepsize * 1e3
        f = numpy.arange(numpy.floor(nt/2)) * propcontrol.Fs * 1e-6 / nt

        nh = propcontrol.harmonic
        dt = propcontrol.dt
        fc = propcontrol.fc
        bw = propcontrol.bandwidth
        filt = propcontrol.filter
        nrmp = numpy.max(numpy.max(numpy.abs(hilbert(u))))

        data = makedb(numpy.transpose(u), dyn, nrmp)
        axs[nh - 1, 0].imshow(data,
                              cmap=cmap,
                              aspect='auto',
                              extent=([numpy.min(t), numpy.max(t), numpy.max(x), numpy.min(x)]))
        axs[nh - 1, 0].set_xlabel('Time [us]')
        axs[nh - 1, 0].set_ylabel('Depth [mm]')
        axs[nh - 1, 0].set_title('Axial pulse, total field')

        U = numpy.fft.fftn(u, axes=(0,))
        nrmf = numpy.max(numpy.max(numpy.abs(U)))
        data = makedb(numpy.transpose(numpy.abs(U[:int(numpy.floor(nt/2)), :])), dyn, nrmf)
        axs[nh - 1, 1].imshow(data,
                              cmap=cmap,
                              aspect='auto',
                              extent=([numpy.min(f), numpy.max(f), numpy.max(x), numpy.min(x)]))
        axs[nh - 1, 1].set_xlabel('Frequency [MHz]')
        axs[nh - 1, 1].set_ylabel('Depth [mm]')
        axs[nh - 1, 1].set_title('Frequency spectrum, total field')

        for ii in range(nh):
            idx = ii + 1
            tmp, _ = bandpass(u, idx * fc, dt, idx * filt * bw, 4)
            tmp = numpy.squeeze(tmp)
            data = makedb(numpy.transpose(tmp), dyn, nrmp)
            axs[idx, 0].imshow(data,
                               cmap=cmap,
                               aspect='auto',
                               extent=([numpy.min(t), numpy.max(t), numpy.max(x), numpy.min(x)]))
            axs[idx, 0].set_xlabel('Time [us]')
            axs[idx, 0].set_ylabel('Depth [mm]')
            axs[idx, 0].set_title('Axial pulse, {}. harmonic'.format(idx))
            U = numpy.fft.fftn(tmp, axes=(0,))
            data = makedb(numpy.transpose(numpy.abs(U[:int(numpy.floor(nt/2)), :])), dyn, nrmf)
            axs[idx, 1].imshow(data,
                               cmap=cmap,
                               aspect='auto',
                               extent=([numpy.min(f), numpy.max(f), numpy.max(x), numpy.min(x)]))
            axs[idx, 1].set_xlabel('Frequency [MHz]')
            axs[idx, 1].set_ylabel('Depth [mm]')
            axs[idx, 1].set_title('Frequency spectrum, {}. harmonic'.format(idx))