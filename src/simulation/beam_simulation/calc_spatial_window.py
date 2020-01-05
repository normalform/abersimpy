"""
calc_spatial_window.py

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
from simulation.beam_simulation.get_window import get_window


def calc_spatial_window(control,
                        window,
                        annular_transducer,
                        num_points_x,
                        num_points_y,
                        resolution_x,
                        resolution_y,
                        step_size):
    if window is None:
        _window = control.simulation.num_windows
    else:
        _window = window
    if isinstance(_window, int) and _window > 0:
        _window = get_window((num_points_x, num_points_y),
                             (resolution_x, resolution_y),
                             _window * step_size,
                             2 * step_size,
                             annular_transducer)

    return _window
