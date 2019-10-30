import matplotlib.pyplot as plt
import numpy

from consts import ProfileHistory, NoAberrationAndHomogeneousMedium
from prop_control import PropControl, Config, ExactDiffraction
from simscript.beamsim import beamsim
from transducer.pulsegenerator import pulsegenerator
from visualization.plot_beamprofile import plot_beamprofile
from visualization.plot_pulse import plot_pulse

if __name__ == '__main__':
    # input variables for Propcontrol
    name = 'test_2dLinHom'
    num_dimensions = 2
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

    # generate Propcontrol
    prop_control = PropControl.init_prop_control(name, num_dimensions, config, harm)

    # generate a  wave field at the transducer
    u, _, _ = pulsegenerator(prop_control, 'transducer')

    # running the simulation
    u_out, prop_control, rmspro, maxpro, axplse, _ = beamsim(prop_control, u)

    # visualization of results
    plot_beamprofile(rmspro, prop_control)

    # find index of focal profile
    idx = int(numpy.round(prop_control.Fx / prop_control.stepsize))
    plot_beamprofile(rmspro[..., idx:idx + 1, :], prop_control)

    plot_pulse(axplse, prop_control, 1)

    plt.show()
