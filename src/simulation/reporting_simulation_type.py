"""
reporting_simulation_type.py
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
