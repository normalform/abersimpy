"""
estimate_eta.py

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
import math
import time
from typing import List

_TEN_PERCENT = 10
_TIME_TO_SHOW = 10
_MINUTE = 60
_HOUR = _MINUTE * 60
_DECIMAL = 10
_DAY_HOUR = 24
_PERCENT = 100


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
        _type = 'to end of body wall'
    else:
        _type = 'to final end point'
    step_index = step_index + 1
    _ten_percent = (step_index % math.ceil(num_steps / _TEN_PERCENT)) == 0
    if _ten_percent or times[step_index] - lap_time > _TIME_TO_SHOW:
        _progress = float(step_index) / float(num_steps)
        _current_time = time.localtime()
        _lap_time = times[step_index]
        _estimated_total_time = times[step_index] / _progress
        _eta = times[step_index] - _estimated_total_time
        _eth = _current_time.tm_hour + math.floor(-_eta / _HOUR)
        _eta = _eta + math.floor(-_eta / _HOUR) * _HOUR
        _etm = _current_time.tm_min + math.floor(-_eta / _MINUTE)

        if _etm >= _MINUTE:
            _etm = _etm % _MINUTE
        if _etm < _DECIMAL:
            _minutes_str = '{:1d}'.format(int(_etm))
        else:
            _minutes_str = '{:2d}'.format(int(_etm))
        if _eth >= _DAY_HOUR:
            _day_str = '+{:2d}'.format(int(math.floor(_eth / _DAY_HOUR)))
            _eth = _eth % _DAY_HOUR
        else:
            _day_str = ''
        if _eth < _DECIMAL:
            _hours_str = '{:1d}'.format(int(_eth))
        else:
            _hours_str = '{:2d}'.format(int(_eth))
        _time_str = '{}:{} {}'.format(_hours_str, _minutes_str, _day_str)

        print('Simulation {} {:2.2f} % progress ({}/{}), ETA {}'.format(_type,
                                                                        _PERCENT * _progress,
                                                                        step_index,
                                                                        num_steps,
                                                                        _time_str))
    else:
        _lap_time = lap_time

    return _lap_time
