import matplotlib.pyplot as plt
import numpy

from controls.consts import NoAberrationAndHomogeneousMedium
from controls.main_control import MainControl
from diffraction.diffraction import ExactDiffraction
from simscript.beam_simulation import beam_simulation
from transducer.pulsegenerator import pulsegenerator
from visualization.plot_beamprofile import plot_beamprofile
from visualization.plot_pulse import plot_pulse

if __name__ == '__main__':
    # generate control
    control = MainControl(simulation_name='test_3dLinHom',
                          num_dimensions=3,
                          diffraction_type=ExactDiffraction,
                          non_linearity=False,
                          attenuation=True,
                          heterogeneous_medium=NoAberrationAndHomogeneousMedium,
                          harmonic=1)

    # generate a  wave field at the transducer
    u, _, _ = pulsegenerator(control, 'transducer', [0, 1])

    # 3D pulses may also be plotted
    plot_pulse(u, control)

    # running the simulation
    u_out, rms_pro, max_pro, ax_pulse, _ = beam_simulation(control, u)

    # visualization of results
    plot_beamprofile(rms_pro, control)

    # find index of focal profile
    idx = int(numpy.round(control.transducer.focus_azimuth / control.simulation.step_size))
    plot_beamprofile(rms_pro[..., idx:idx + 1, :], control)

    plot_pulse(ax_pulse, control, 1)

    plt.show()
