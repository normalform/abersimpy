# -*- coding: utf-8 -*-
"""
    reporting_simulation_type.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""


def reporting_simulation_type(non_linearity,
                              num_dimensions,
                              num_points_t,
                              num_points_x,
                              num_points_y):
    if non_linearity:
        non_linearity_str = 'non-linear'
    else:
        non_linearity_str = 'linear'
    if num_dimensions == 2:
        dimensions_str = f'{num_points_x}X * {num_points_t}T'
    else:
        dimensions_str = f'{num_points_x}X * {num_points_y}Y * {num_points_t}T'

    print(f'Starting {non_linearity_str} simulation of size {dimensions_str}')
