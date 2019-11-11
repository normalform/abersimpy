"""
plot_pulse.py
"""
import matplotlib.pyplot as plt
import numpy
from scipy.signal import hilbert

from filter.bandpass import bandpass
from visualization.make_db import make_db


def plot_pulse(u,
               control,
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

    center_channel = control.transducer.center_channel
    t = numpy.arange(-numpy.floor(num_points_t / 2),
                     numpy.ceil(num_points_t / 2)) * control.signal.resolution_t * 1e6
    y = numpy.arange(-numpy.floor(num_points_y / 2),
                     numpy.ceil(num_points_y / 2)) * control.signal.resolution_y * 1e3

    fig, axs = plt.subplots(2, 2)
    fig.canvas.set_window_title('Pulse')
    if axflag == 0:
        x = numpy.arange(-numpy.floor(num_points_x / 2),
                         numpy.ceil(num_points_x / 2)) * control.signal.resolution_x * 1e3
        if control.annular_transducer and control.num_dimensions == 2:
            x = numpy.arange(num_points_x) * control.signal.resolution_x * 1e3

        if num_dimensions == 3:
            data = make_db(numpy.transpose(numpy.squeeze(u[:, int(center_channel[1]), :])), dyn,
                           nrm)
            axs[0, 0].imshow(data,
                             cmap=cmap,
                             aspect='auto',
                             extent=([numpy.min(t), numpy.max(t), numpy.max(x), numpy.min(x)]))
            axs[0, 0].set_xlabel('Time [us]')
            axs[0, 0].set_ylabel('Azimuth [mm]')

            axs[0, 1].plot(numpy.squeeze(u[:, int(center_channel[1]), int(center_channel[0])]))
            axs[0, 1].set_xlabel('Time [us]')
            axs[0, 1].set_ylabel('Pressure [MPa]')

            data = make_db(numpy.transpose(numpy.squeeze(u[:, :, int(center_channel[0])])), dyn,
                           nrm)
            axs[1, 0].imshow(data,
                             cmap=cmap,
                             aspect='auto',
                             extent=([numpy.min(t), numpy.max(t), numpy.max(x), numpy.min(x)]))
            axs[1, 0].set_xlabel('Time [us]')
            axs[1, 0].set_ylabel('Elevation [mm]')

            data = make_db(numpy.squeeze(numpy.sqrt(numpy.sum(u ** 2, axis=0))), dyn, nrm)
            axs[1, 1].imshow(data,
                             cmap=cmap,
                             aspect='auto',
                             extent=([numpy.min(x), numpy.max(x), numpy.min(y), numpy.max(y)]))
            axs[1, 1].set_xlabel('Azimuth [mm]')
            axs[1, 1].set_ylabel('Elevation [mm]')
        else:
            raise NotImplementedError
    else:
        x = numpy.arange(num_points_x) * control.simulation.step_size * 1e3
        f = numpy.arange(
            numpy.floor(num_points_t / 2)) * control.signal.sample_frequency * 1e-6 / num_points_t

        nh = control.harmonic
        resolution_t = control.signal.resolution_t
        transmit_frequency = control.signal.transmit_frequency
        bw = control.signal.bandwidth
        filt = control.signal.filter
        nrmp = numpy.max(numpy.max(numpy.abs(hilbert(u))))

        data = make_db(numpy.transpose(u), dyn, nrmp)
        axs[nh - 1, 0].imshow(data,
                              cmap=cmap,
                              aspect='auto',
                              extent=([numpy.min(t), numpy.max(t), numpy.max(x), numpy.min(x)]))
        axs[nh - 1, 0].set_xlabel('Time [us]')
        axs[nh - 1, 0].set_ylabel('Depth [mm]')
        axs[nh - 1, 0].set_title('Axial pulse, total field')

        U = numpy.fft.fftn(u, axes=(0,))
        nrmf = numpy.max(numpy.max(numpy.abs(U)))
        data = make_db(numpy.transpose(numpy.abs(U[:int(numpy.floor(num_points_t / 2)), :])), dyn,
                       nrmf)
        axs[nh - 1, 1].imshow(data,
                              cmap=cmap,
                              aspect='auto',
                              extent=([numpy.min(f), numpy.max(f), numpy.max(x), numpy.min(x)]))
        axs[nh - 1, 1].set_xlabel('Frequency [MHz]')
        axs[nh - 1, 1].set_ylabel('Depth [mm]')
        axs[nh - 1, 1].set_title('Frequency spectrum, total field')

        for ii in range(nh):
            idx = ii + 1
            tmp, _ = bandpass(u,
                              numpy.array([idx * transmit_frequency]),
                              resolution_t,
                              idx * filt * bw,
                              4)
            tmp = numpy.squeeze(tmp)
            data = make_db(numpy.transpose(tmp), dyn, nrmp)
            axs[idx, 0].imshow(data,
                               cmap=cmap,
                               aspect='auto',
                               extent=([numpy.min(t), numpy.max(t), numpy.max(x), numpy.min(x)]))
            axs[idx, 0].set_xlabel('Time [us]')
            axs[idx, 0].set_ylabel('Depth [mm]')
            axs[idx, 0].set_title('Axial pulse, {}. harmonic'.format(idx))
            U = numpy.fft.fftn(tmp, axes=(0,))
            data = make_db(numpy.transpose(numpy.abs(U[:int(numpy.floor(num_points_t / 2)), :])),
                           dyn, nrmf)
            axs[idx, 1].imshow(data,
                               cmap=cmap,
                               aspect='auto',
                               extent=([numpy.min(f), numpy.max(f), numpy.max(x), numpy.min(x)]))
            axs[idx, 1].set_xlabel('Frequency [MHz]')
            axs[idx, 1].set_ylabel('Depth [mm]')
            axs[idx, 1].set_title('Frequency spectrum, {}. harmonic'.format(idx))
