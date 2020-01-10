# -*- coding: utf-8 -*-
"""
    adjust_equidistant_steps.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
from typing import Tuple, List, Type, Union

from system.diffraction.diffraction import ExactDiffraction, PseudoDifferential
from system.diffraction.interfaces import IDiffractionType


def adjust_equidistant_steps(diffraction_type: Union[IDiffractionType, Type[IDiffractionType]],
                             equidistant_steps: bool,
                             num_steps: int,
                             step_index: List[int]) -> Tuple[List[int], bool, bool]:
    diff_step_idx = [0] + list(differences(step_index))
    num_recalculate = get_num_recalculate(diff_step_idx)
    if diffraction_type in (ExactDiffraction, PseudoDifferential):
        if diffraction_type is ExactDiffraction:
            _diffraction_factor = 1
        else:
            _diffraction_factor = 3
        if num_recalculate / num_steps < 0.5 / _diffraction_factor:
            recalculate = True
            if step_index[0] == 0:
                _equidistant_steps = True
            else:
                _equidistant_steps = equidistant_steps
        else:
            recalculate = False
            _equidistant_steps = False
    else:
        recalculate = False
        _equidistant_steps = equidistant_steps

    return diff_step_idx, _equidistant_steps, recalculate


def get_num_recalculate(seq):
    return sum(1 for i in seq if i > 0)


# TODO Fix PyLint warning
def differences(seq):
    _iterable = iter(seq)
    _prev = next(_iterable)
    for element in _iterable:
        yield element - _prev
        _prev = element
