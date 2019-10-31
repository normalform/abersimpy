import matplotlib.pyplot as plt
import numpy

from controls.consts import ProfileHistory, NoAberrationAndHomogeneousMedium
from controls.prop_control import PropControl, Config, ExactDiffraction
from simscript.beam_simulation import beam_simulation
from transducer.pulsegenerator import pulsegenerator
from visualization.plot_beamprofile import plot_beamprofile
from visualization.plot_pulse import plot_pulse

if __name__ == '__main__':
    # input variables for PropControl
    name = 'test_3dLinHom'
    num_dimensions = 3
    config = Config(
        diffraction_type=ExactDiffraction,
        non_linearity=False,
        attenuation=True,
        heterogeneous_medium=NoAberrationAndHomogeneousMedium,
        annular_transducer=False,
        equidistant_steps=False,
        history=ProfileHistory
    )
    harm = 1

    # generate PropControl
    prop_control = PropControl(name, num_dimensions, config, harm)

    # generate a  wave field at the transducer
    u, _, _ = pulsegenerator(prop_control, 'transducer', [0, 1])

    # 3D pulses may also be plotted
    plot_pulse(u, prop_control)

    # running the simulation
    u_out, prop_control, rmspro, maxpro, axplse, _ = beam_simulation(prop_control, u)

    # visualization of results
    plot_beamprofile(rmspro, prop_control)

    # find index of focal profile
    idx = int(numpy.round(prop_control.focus_azimuth / prop_control.step_size))
    plot_beamprofile(rmspro[..., idx:idx + 1, :], prop_control)

    plot_pulse(axplse, prop_control, 1)

    plt.show()
