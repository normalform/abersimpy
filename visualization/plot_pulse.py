import matplotlib.pyplot as plt
import numpy
from scipy.signal import hilbert

from filter.bandpass import bandpass
from visualization.makedb import makedb


def plot_pulse(u,
               main_control,
               axflag=0,
               dyn=40,
               nrm=None,
               fh=None,
               cmap='gray'):
    num_dimensions = u.ndim
    num_points_t = u.shape[0]
    num_points_y = u.shape[1]
    if num_dimensions == 2:
        num_points_x = num_points_y
    else:
        num_points_x = u.shape[2]

    cc = main_control.center_channel
    t = numpy.arange(-numpy.floor(num_points_t / 2), numpy.ceil(num_points_t / 2)) * main_control.resolution_t * 1e6
    y = numpy.arange(-numpy.floor(num_points_y / 2), numpy.ceil(num_points_y / 2)) * main_control.resolution_y * 1e3

    fig, axs = plt.subplots(2, 2)
    fig.canvas.set_window_title('Pulse')
    if axflag == 0:
        x = numpy.arange(-numpy.floor(num_points_x / 2), numpy.ceil(num_points_x / 2)) * main_control.resolution_x * 1e3
        if main_control.config.annular_transducer and main_control.num_dimensions == 2:
            x = numpy.arange(num_points_x) * main_control.resolution_x * 1e3

        if num_dimensions == 3:
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
        x = numpy.arange(num_points_x) * main_control.step_size * 1e3
        f = numpy.arange(numpy.floor(num_points_t / 2)) * main_control.sample_frequency * 1e-6 / num_points_t

        nh = main_control.harmonic
        resolution_t = main_control.resolution_t
        transmit_frequency = main_control.transmit_frequency
        bw = main_control.bandwidth
        filt = main_control.filter
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
        data = makedb(numpy.transpose(numpy.abs(U[:int(numpy.floor(num_points_t / 2)), :])), dyn, nrmf)
        axs[nh - 1, 1].imshow(data,
                              cmap=cmap,
                              aspect='auto',
                              extent=([numpy.min(f), numpy.max(f), numpy.max(x), numpy.min(x)]))
        axs[nh - 1, 1].set_xlabel('Frequency [MHz]')
        axs[nh - 1, 1].set_ylabel('Depth [mm]')
        axs[nh - 1, 1].set_title('Frequency spectrum, total field')

        for ii in range(nh):
            idx = ii + 1
            tmp, _ = bandpass(u, idx * transmit_frequency, resolution_t, idx * filt * bw, 4)
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
            data = makedb(numpy.transpose(numpy.abs(U[:int(numpy.floor(num_points_t / 2)), :])), dyn, nrmf)
            axs[idx, 1].imshow(data,
                               cmap=cmap,
                               aspect='auto',
                               extent=([numpy.min(f), numpy.max(f), numpy.max(x), numpy.min(x)]))
            axs[idx, 1].set_xlabel('Frequency [MHz]')
            axs[idx, 1].set_ylabel('Depth [mm]')
            axs[idx, 1].set_title('Frequency spectrum, {}. harmonic'.format(idx))
