from initpropcontrol import initpropcontrol
from transducer.pulsegenerator import pulsegenerator
from simscript.beamsim import beamsim
from visualization.plot_beamprofile import plot_beamprofile
from visualization.plot_pulse import plot_pulse

import numpy


if __name__ == '__main__':
    # input variables for Propcontrol
    name = 'test_2dlinhom'
    ndim = 2
    flags = [1, 0, 1, 0, 0, 0, 2]
    harm = 1

    # generate Propcontrol
    propcontrol = initpropcontrol(name, ndim, flags, harm)

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
