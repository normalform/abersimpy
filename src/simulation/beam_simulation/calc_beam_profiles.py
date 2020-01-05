"""
calc_beam_profiles.py

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
import numpy

from simulation.controls.consts import NoHistory
from simulation.post_processing.export_beam_profile import export_beam_profile


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
