# -*- coding: utf-8 -*-
"""
    recalculate_wave_numbers.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
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
