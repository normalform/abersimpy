"""
calc_spatial_window.py
"""
from simulation.beam_simulation.get_window import get_window


def calc_spatial_window(control,
                        window,
                        annular_transducer,
                        num_points_x,
                        num_points_y,
                        resolution_x,
                        resolution_y,
                        step_size):
    if window is None:
        _window = control.simulation.num_windows
    else:
        _window = window
    if isinstance(_window, int) and _window > 0:
        _window = get_window((num_points_x, num_points_y),
                             (resolution_x, resolution_y),
                             _window * step_size,
                             2 * step_size,
                             annular_transducer)

    return _window
