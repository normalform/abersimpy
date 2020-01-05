"""
adjust_equidistant_steps.py

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

from system.diffraction.diffraction import ExactDiffraction, PseudoDifferential


def adjust_equidistant_steps(control,
                             equidistant_steps,
                             num_steps,
                             step_idx):
    _diff_step_idx = numpy.concatenate((numpy.array([0], dtype=int), numpy.diff(step_idx)))
    _num_recalculate = len(numpy.where(_diff_step_idx)[0])
    if control.diffraction_type == ExactDiffraction or \
            control.diffraction_type == PseudoDifferential:
        if control.diffraction_type == ExactDiffraction:
            _diffraction_factor = 1
        else:
            _diffraction_factor = 3
        if _num_recalculate / num_steps < 0.5 / _diffraction_factor:
            _recalculate = True
            if step_idx[0] == 0:
                _equidistant_steps = True
            else:
                _equidistant_steps = equidistant_steps
        else:
            _recalculate = False
            _equidistant_steps = False
    else:
        _recalculate = False
        _equidistant_steps = equidistant_steps

    return _diff_step_idx, _equidistant_steps, _recalculate
