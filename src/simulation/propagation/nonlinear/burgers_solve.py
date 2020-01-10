# -*- coding: utf-8 -*-
"""
    burgers_solve.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import numpy


def burgers_solve(time_span,
                  pressure,
                  permutation,
                  eps_n,
                  resolution_z):
    """
    Non-linear distortion of periodic pressure pulse with original time sampling.
    :param time_span: Scaled time span.
    :param pressure: Pressure.
    :param permutation: Permutation to be introduced.
    :param eps_n: Coefficient of non-linearity.
    :param resolution_z: Step size.
    :return: Perturbed wave field.
    """
    num_points_t = numpy.max(time_span.shape)
    resolution_t = time_span[1] - time_span[0]
    num_points = (time_span[num_points_t - 1] - time_span[0]) + resolution_t

    # introduce permutation
    t2 = time_span - eps_n * resolution_z * permutation

    # extends by periodicity
    idt = int(numpy.floor(num_points_t / 10))
    idx_tail = numpy.arange(idt)
    idx_front = numpy.arange(num_points_t - idt, num_points_t)

    t_tail = t2[idx_tail] + num_points
    t_front = t2[idx_front] - num_points
    t2 = numpy.concatenate((t_front, t2, t_tail))

    pressure_tail = pressure[idx_tail]
    pressure_front = pressure[idx_front]
    pressure2 = numpy.concatenate((pressure_front, pressure, pressure_tail))

    # re-sample at equidistant time points
    _pressure = _get_linear(t2, pressure2, time_span)

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
    num_t2 = numpy.max(equidistant_time_span.shape)
    index_t2 = 2

    pressure_values_t2 = []
    for index in range(num_t2):
        while equidistant_time_span[index] > time_span[index_t2]:
            index_t2 = index_t2 + 1
        index_t1 = index_t2 - 1
        dp_dt = (pressure_values[index_t2] - pressure_values[index_t1]) / \
                (time_span[index_t2] - time_span[index_t1])
        pressure_values_t2.append(
            pressure_values[index_t1] +
            (equidistant_time_span[index] - time_span[index_t1]) * dp_dt)

    return numpy.array(pressure_values_t2)
