# -*- coding: utf-8 -*-
"""
    estimate_eta.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import math
import time
from typing import List

TEN_PERCENT = 10
TIME_TO_SHOW = 10
MINUTE = 60
HOUR = MINUTE * 60
DECIMAL = 10
DAY_HOUR = 24
PERCENT = 100


def estimate_eta(times: List[float],
                 num_steps: int,
                 step_index: int,
                 lap_time: float,
                 body_wall: bool = False):
    """
    TODO Might need better design
    :param times:
    :param num_steps:
    :param step_index:
    :param lap_time:
    :param body_wall:
    :return:
    """
    if body_wall:
        type = 'to end of body wall'
    else:
        type = 'to final end point'
    step_index = step_index + 1
    ten_percent = (step_index % math.ceil(num_steps / TEN_PERCENT)) == 0
    if ten_percent or times[step_index] - lap_time > TIME_TO_SHOW:
        progress = float(step_index) / float(num_steps)
        current_time = time.localtime()
        lap_time = times[step_index]
        estimated_total_time = times[step_index] / progress
        eta = times[step_index] - estimated_total_time
        eth = current_time.tm_hour + math.floor(-eta / HOUR)
        eta = eta + math.floor(-eta / HOUR) * HOUR
        etm = current_time.tm_min + math.floor(-eta / MINUTE)

        if etm >= MINUTE:
            etm = etm % MINUTE
        if etm < DECIMAL:
            minutes_str = '{:1d}'.format(int(etm))
        else:
            minutes_str = '{:2d}'.format(int(etm))
        if eth >= DAY_HOUR:
            day_str = '+{:2d}'.format(int(math.floor(eth / DAY_HOUR)))
            eth = eth % DAY_HOUR
        else:
            day_str = ''
        if eth < DECIMAL:
            hours_str = '{:1d}'.format(int(eth))
        else:
            hours_str = '{:2d}'.format(int(eth))
        time_str = '{}:{} {}'.format(hours_str, minutes_str, day_str)

        print('Simulation {} {:2.2f} % progress ({}/{}), ETA {}'.format(type,
                                                                        PERCENT * progress,
                                                                        step_index,
                                                                        num_steps,
                                                                        time_str))
    else:
        lap_time = lap_time

    return lap_time
