"""
Tutorial for simulation of 2D non-linear homogeneous case

Copyright (C) 2020  Jaeho Kim

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import matplotlib.pyplot as plt
import numpy

from simulation.controls.consts import NoAberrationAndHomogeneousMedium
from simulation.controls.main_control import MainControl
from simulation.simulation import simulation
from system.diffraction.diffraction import ExactDiffraction
from system.transducer.pulse_generator import pulse_generator
from visualization import plot_beam_profile
from visualization import plot_pulse

if __name__ == '__main__':
    # generate control
    control = MainControl(simulation_name='test_2dNonlinHom',
                          num_dimensions=2,
                          diffraction_type=ExactDiffraction,
                          non_linearity=True,
                          attenuation=True,
                          heterogeneous_medium=NoAberrationAndHomogeneousMedium,
                          harmonic=1)

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
