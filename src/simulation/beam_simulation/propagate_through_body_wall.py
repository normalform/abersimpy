# -*- coding: utf-8 -*-
"""
    propagate_through_body_wall.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
from simulation.beam_simulation.body_wall import body_wall
from simulation.controls.consts import NO_HISTORY


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

        if history != NO_HISTORY:
            pnx, pny, pns, pnh = _rms_pro.shape
            step_index = pns
            raise NotImplementedError
    else:
        _ax_pulse = ax_pulse
        _max_pro = max_pro
        _rms_pro = rms_pro
        _wave_field = wave_field
        _z_pos = z_pos

    return _ax_pulse, _max_pro, _rms_pro, _wave_field, _z_pos
