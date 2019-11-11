"""
calc_beam_profiles.py
"""
import numpy

from controls.consts import NoHistory
from postprocessing.export_beam_profile import export_beam_profile


def calc_beam_profiles(control,
                       history,
                       num_points_t,
                       num_points_x,
                       num_points_y,
                       num_steps,
                       step_index,
                       wave_field):
    if history != NoHistory:
        _rms_pro = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        _max_pro = numpy.zeros((num_points_y, num_points_x, num_steps, control.harmonic + 1))
        _ax_pulse = numpy.zeros((num_points_t, num_steps))
        _z_pos = numpy.zeros(num_steps)
    else:
        _rms_pro = numpy.array([])
        _max_pro = numpy.array([])
        _ax_pulse = numpy.array([])
        _z_pos = numpy.array([])

    _step_index = 0

    _rms_pro, _max_pro, _ax_pulse, _z_pos = export_beam_profile(control,
                                                                wave_field,
                                                                _rms_pro,
                                                                _max_pro,
                                                                _ax_pulse,
                                                                _z_pos,
                                                                step_index)

    return _ax_pulse, _max_pro, _rms_pro, _z_pos
