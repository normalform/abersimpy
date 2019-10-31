import matplotlib.pyplot as plt
import numpy

from controls.consts import ProfileHistory, NoAberrationAndHomogeneousMedium
from controls.main_control import MainControl, ConfigControl, ExactDiffraction
from simscript.beam_simulation import beam_simulation
from transducer.pulsegenerator import pulsegenerator
from visualization.plot_beamprofile import plot_beamprofile
from visualization.plot_pulse import plot_pulse

if __name__ == '__main__':
    # input variables for MainControl
    name = 'test_3dLinHom'
    num_dimensions = 3
    config = ConfigControl(
        diffraction_type=ExactDiffraction,
        non_linearity=False,
        attenuation=True,
        heterogeneous_medium=NoAberrationAndHomogeneousMedium,
        annular_transducer=False,
        equidistant_steps=False,
        history=ProfileHistory
    )
    harm = 1

    # generate MainControl
    main_control = MainControl(name, num_dimensions, config, harm)

    # generate a  wave field at the transducer
    u, _, _ = pulsegenerator(main_control, 'transducer', [0, 1])

    # 3D pulses may also be plotted
    plot_pulse(u, main_control)

    # running the simulation
    u_out, main_control, rmspro, maxpro, axplse, _ = beam_simulation(main_control, u)

    # visualization of results
    plot_beamprofile(rmspro, main_control)

    # find index of focal profile
    idx = int(numpy.round(main_control.transducer.focus_azimuth / main_control.simulation.step_size))
    plot_beamprofile(rmspro[..., idx:idx + 1, :], main_control)

    plot_pulse(axplse, main_control, 1)

    plt.show()
