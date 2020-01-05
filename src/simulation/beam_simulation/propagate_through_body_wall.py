"""
propagate_through_body_wall.py

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
from simulation.beam_simulation.body_wall import body_wall
from simulation.controls.consts import NoHistory


def propagate_through_body_wall(control,
                                phantom,
                                wave_numbers,
                                ax_pulse,
                                current_pos,
                                equidistant_steps,
                                history,
                                max_pro,
                                rms_pro,
                                wave_field,
                                window,
                                z_pos):
    if control.heterogeneous_medium != 0 and current_pos < control.material.thickness:
        print('Entering body wall')
        _wave_field, _rms_pro, _max_pro, _ax_pulse, _z_pos = body_wall(control,
                                                                       wave_field,
                                                                       1,
                                                                       equidistant_steps,
                                                                       wave_numbers,
                                                                       window,
                                                                       phantom)
        print('Done with body wall')

        if history != NoHistory:
            _pnx, _pny, _pns, _pnh = _rms_pro.shape
            _step_index = _pns
            raise NotImplementedError
    else:
        _ax_pulse = ax_pulse
        _max_pro = max_pro
        _rms_pro = rms_pro
        _wave_field = wave_field
        _z_pos = z_pos

    return _ax_pulse, _max_pro, _rms_pro, _wave_field, _z_pos
