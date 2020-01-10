# -*- coding: utf-8 -*-
"""
    get_transducer_indexes.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import numpy

from simulation.controls.main_control import MainControl


def get_transducer_indexes(control: MainControl):
    """
    Function that returns the indices in x and y covered by the transducer surface.
    :param control: The control.
    :return: Indices for the transducer surface in the x-direction.
             Indices for the transducer surface in the y-direction.
             The indices for the area outside the transducer in x.
             The indices for the area outside the transducer in y.
             The center channel for the transducer field.
    """
    # initiation of parameters
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    resolution_x = control.signal.resolution_x
    resolution_y = control.signal.resolution_y
    num_element_x = control.transducer.num_elements_azimuth
    num_element_y = control.transducer.num_elements_elevation
    elements_size_azimuth = control.transducer.elements_size_azimuth
    elements_size_elevation = control.transducer.elements_size_elevation
    center_channel = control.transducer.center_channel
    num_dimensions = control.num_dimensions
    annular_transducer = control.annular_transducer

    # set lengths of transducers
    if num_dimensions == 1:
        index_xs = 1
        index_ys = 1
        index_x0 = []
        index_y0 = []
        ccs = [1, 1]
        return index_xs, index_ys, index_x0, index_y0, ccs

    if annular_transducer and num_dimensions == 2:
        ndx = numpy.round((2 * num_element_x - 1) * elements_size_azimuth / resolution_x)
        ndy = 1
    elif annular_transducer:
        ndx = numpy.round((2 * num_element_x - 1) * elements_size_azimuth / resolution_x)
        ndy = numpy.round((2 * num_element_y - 1) * elements_size_elevation / resolution_y)
    else:
        ndx = numpy.round(num_element_x * elements_size_azimuth / resolution_x)
        ndy = numpy.round(num_element_y * elements_size_elevation / resolution_y)

    if num_dimensions == 2:
        ndy = 1

    # set up indices
    if annular_transducer and num_dimensions == 2:
        index_xs = numpy.arange(0, numpy.ceil(ndx / 2)) + center_channel[0]
        index_ys = numpy.array(1)
    else:
        index_xs = numpy.arange(-numpy.floor(ndx / 2), numpy.ceil(ndx / 2)) + center_channel[0]
        index_ys = numpy.arange(-numpy.floor(ndy / 2), numpy.ceil(ndy / 2)) + center_channel[1]

    index_x0 = numpy.setxor1d(index_xs, numpy.arange(1, num_points_x + 1))
    index_y0 = numpy.setxor1d(index_ys, numpy.arange(1, num_points_y + 1))

    # set center index for transducer field
    ccs = numpy.zeros((2, 1))
    if annular_transducer and num_dimensions == 2:
        ccs[0] = 1
        ccs[1] = 1
    else:
        ccs[0] = numpy.floor(ndx / 2) + 1
        ccs[1] = numpy.floor(ndy / 2) + 1
        if center_channel[1] <= 1:
            center_channel[1] = 1

    return index_xs.astype(int) - 1, \
           index_ys.astype(int) - 1, \
           index_x0.astype(int) - 1, \
           index_y0.astype(int) - 1, \
           ccs.astype(int) - 1
