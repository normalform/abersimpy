# -*- coding: utf-8 -*-
"""
    Example for simulation of 2D non-linear homogeneous case
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import matplotlib.pyplot as plt
import numpy

from simulation.controls.consts import NO_ABERRATION_AND_HOMOGENEOUS_MEDIUM
from simulation.controls.main_control import MainControl
from simulation.simulation import simulation
from system.diffraction.diffraction import ExactDiffraction
from system.transducer.pulse_generator import pulse_generator
from visualization.plot_beam_profile import plot_beam_profile
from visualization.plot_pulse import plot_pulse

if __name__ == '__main__':
    # generate control
    control = MainControl(simulation_name='test_2dNonlinHom',
                          num_dimensions=2,
                          diffraction_type=ExactDiffraction,
                          non_linearity=True,
                          attenuation=True,
                          heterogeneous_medium=NO_ABERRATION_AND_HOMOGENEOUS_MEDIUM,
                          harmonic=1,
                          end_point=0.01,
                          focus_azimuth=0.005,
                          focus_elevation=0.005)

    # generate a  wave field at the transducer
    pulse, _ = pulse_generator(control, 'transducer')

    # running the simulation
    wave_field, rms_profile, max_profile, ax_pulse, _ = simulation(control, pulse)

    # visualization of results
    plot_beam_profile(rms_profile, control)

    # find index of focal profile
    idx = int(numpy.round(control.transducer.focus_azimuth / control.simulation.step_size))
    plot_beam_profile(rms_profile[..., idx:idx + 1, :], control)

    plot_pulse(ax_pulse, control, 1)

    plt.show()
