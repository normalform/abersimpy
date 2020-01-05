"""
reporting_simulation_type.py

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


def reporting_simulation_type(non_linearity,
                              num_dimensions,
                              num_points_t,
                              num_points_x,
                              num_points_y):
    if non_linearity:
        _non_linearity_str = 'non-linear'
    else:
        _non_linearity_str = 'linear'
    if num_dimensions == 2:
        _dimensions_str = f'{num_points_x}X * {num_points_t}T'
    else:
        _dimensions_str = f'{num_points_x}X * {num_points_y}Y * {num_points_t}T'

    print(f'Starting {_non_linearity_str} simulation of size {_dimensions_str}')
