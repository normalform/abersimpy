"""
recalculate_wave_numbers.py

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
from simulation.get_wave_numbers import get_wave_numbers


def recalculate_wave_numbers(control,
                             wave_numbers,
                             diff_step_idx,
                             equidistant_steps,
                             index,
                             recalculate,
                             step_idx):
    if recalculate:
        if diff_step_idx[index] != 0:
            if step_idx[index] == 0:
                _equidistant_steps = False
            else:
                _equidistant_steps = True
            _wave_numbers = get_wave_numbers(control, _equidistant_steps)
        else:
            _equidistant_steps = equidistant_steps
            _wave_numbers = wave_numbers
    else:
        _equidistant_steps = equidistant_steps
        _wave_numbers = wave_numbers

    return _wave_numbers, _equidistant_steps
