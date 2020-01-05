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
from typing import Tuple, List, Type, Union

from system.diffraction.diffraction import ExactDiffraction, PseudoDifferential
from system.diffraction.interfaces import IDiffractionType


def adjust_equidistant_steps(diffraction_type: Union[IDiffractionType, Type[IDiffractionType]],
                             equidistant_steps: bool,
                             num_steps: int,
                             step_index: List[int]) -> Tuple[List[int], bool, bool]:
    _diff_step_idx = [0] + list(_differences(step_index))
    _num_recalculate = _get_num_recalculate(_diff_step_idx)
    if diffraction_type in (ExactDiffraction, PseudoDifferential):
        if diffraction_type is ExactDiffraction:
            _diffraction_factor = 1
        else:
            _diffraction_factor = 3
        if _num_recalculate / num_steps < 0.5 / _diffraction_factor:
            _recalculate = True
            if step_index[0] == 0:
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


def _get_num_recalculate(seq):
    return sum(1 for i in seq if i > 0)


# TODO Fix PyLint warning
def _differences(seq):
    _iterable = iter(seq)
    _prev = next(_iterable)
    for element in _iterable:
        yield element - _prev
        _prev = element
