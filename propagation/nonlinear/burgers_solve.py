"""
burgers_solve.py
"""
import numpy


def burgers_solve(time_span,
                  pressure,
                  permutation,
                  eps_n,
                  resolution_z):
    """
    Non-linear distortion of periodic pressure puls with original time sampling.
    :param time_span: Scaled time span.
    :param pressure: Pressure.
    :param permutation: Permutation to be introduced.
    :param eps_n: Coefficient of non-linearity.
    :param resolution_z: Step size.
    :return: Perturbed wave field.
    """
    _num_points_t = numpy.max(time_span.shape)
    _resolution_t = time_span[1] - time_span[0]
    _num_points = (time_span[_num_points_t - 1] - time_span[0]) + _resolution_t

    # introduce permutation
    _t2 = time_span - eps_n * resolution_z * permutation

    # extends by periodicity
    _idt = int(numpy.floor(_num_points_t / 10))
    _idx_tail = numpy.arange(_idt)
    _idx_front = numpy.arange(_num_points_t - _idt, _num_points_t)

    _t_tail = _t2[_idx_tail] + _num_points
    _t_front = _t2[_idx_front] - _num_points
    _t2 = numpy.concatenate((_t_front, _t2, _t_tail))

    _pressure_tail = pressure[_idx_tail]
    _pressure_front = pressure[_idx_front]
    _pressure2 = numpy.concatenate((_pressure_front, pressure, _pressure_tail))

    # re-sample at equidistant time points
    _pressure = _get_linear(_t2, _pressure2, time_span)

    return _pressure


def _get_linear(time_span: numpy.ndarray,
                pressure_values: numpy.ndarray,
                equidistant_time_span: numpy.ndarray) -> numpy.ndarray:
    """
    Re-samples a not equidistant t1 with corresponding pressure values p1 to an equidistant time
    span t2 through interpolation.
    :param time_span: Perturbed monotonic time span
    :param pressure_values: Pressure values at t1.
    :param equidistant_time_span: Equidistant monotonic time span.
    :return: Pressure values at t2
    """
    _num_t2 = numpy.max(equidistant_time_span.shape)
    _index_t2 = 2

    _pressure_values_t2 = []
    for _index in range(_num_t2):
        while equidistant_time_span[_index] > time_span[_index_t2]:
            _index_t2 = _index_t2 + 1
        _index_t1 = _index_t2 - 1
        _dpdt = (pressure_values[_index_t2] - pressure_values[_index_t1]) / \
                (time_span[_index_t2] - time_span[_index_t1])
        _pressure_values_t2.append(
            pressure_values[_index_t1] +
            (equidistant_time_span[_index] - time_span[_index_t1]) * _dpdt)

    return numpy.array(_pressure_values_t2)
