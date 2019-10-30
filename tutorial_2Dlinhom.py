import numpy
import matplotlib.pyplot as plt

from initpropcontrol import initpropcontrol
from propcontrol import Config, ExactDiffraction
from transducer.pulsegenerator import pulsegenerator
from simscript.beamsim import beamsim
from visualization.plot_beamprofile import plot_beamprofile
from visualization.plot_pulse import plot_pulse
from consts import PROFHISTORY, NoAberrationAndHomogeneousMedium


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
        history=PROFHISTORY
    )
    harm = 1

    # generate Propcontrol
    propcontrol = initpropcontrol(name, num_dimensions, config, harm)

    # generate a  wave field at the transducer
    u, _, _ = pulsegenerator(propcontrol, 'transducer')

    # running the simulation
    u_out, propcontrol, rmspro, maxpro, axplse, _ = beamsim(propcontrol, u)

    # visualization of results
    plot_beamprofile(rmspro, propcontrol)

    # find index of focal profile
    idx = int(numpy.round(propcontrol.Fx / propcontrol.stepsize))
    plot_beamprofile(rmspro[..., idx:idx+1, :], propcontrol)

    plot_pulse(axplse, propcontrol, 1)

    plt.show()
