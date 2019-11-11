"""
get_transducer_indexes.py
"""
import numpy

from controls.main_control import MainControl


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
    _num_points_x = control.domain.num_points_x
    _num_points_y = control.domain.num_points_y
    _resolution_x = control.signal.resolution_x
    _resolution_y = control.signal.resolution_y
    _num_element_x = control.transducer.num_elements_azimuth
    _num_element_y = control.transducer.num_elements_elevation
    _elements_size_azimuth = control.transducer.elements_size_azimuth
    _elements_size_elevation = control.transducer.elements_size_elevation
    _center_channel = control.transducer.center_channel
    _num_dimensions = control.num_dimensions
    _annular_transducer = control.annular_transducer

    # set lengths of transducers
    if _num_dimensions == 1:
        _index_xs = 1
        _index_ys = 1
        _index_x0 = []
        _index_y0 = []
        _ccs = [1, 1]
        return _index_xs, _index_ys, _index_x0, _index_y0, _ccs

    if _annular_transducer and _num_dimensions == 2:
        _ndx = numpy.round((2 * _num_element_x - 1) * _elements_size_azimuth / _resolution_x)
        _ndy = 1
    elif _annular_transducer:
        _ndx = numpy.round((2 * _num_element_x - 1) * _elements_size_azimuth / _resolution_x)
        _ndy = numpy.round((2 * _num_element_y - 1) * _elements_size_elevation / _resolution_y)
    else:
        _ndx = numpy.round(_num_element_x * _elements_size_azimuth / _resolution_x)
        _ndy = numpy.round(_num_element_y * _elements_size_elevation / _resolution_y)

    if _num_dimensions == 2:
        _ndy = 1

    # set up indices
    if _annular_transducer and _num_dimensions == 2:
        _index_xs = numpy.arange(0, numpy.ceil(_ndx / 2)) + _center_channel[0]
        _index_ys = numpy.array(1)
    else:
        _index_xs = numpy.arange(-numpy.floor(_ndx / 2), numpy.ceil(_ndx / 2)) + _center_channel[0]
        _index_ys = numpy.arange(-numpy.floor(_ndy / 2), numpy.ceil(_ndy / 2)) + _center_channel[1]

    _index_x0 = numpy.setxor1d(_index_xs, numpy.arange(1, _num_points_x + 1))
    _index_y0 = numpy.setxor1d(_index_ys, numpy.arange(1, _num_points_y + 1))

    # set center index for transducer field
    _ccs = numpy.zeros((2, 1))
    if _annular_transducer and _num_dimensions == 2:
        _ccs[0] = 1
        _ccs[1] = 1
    else:
        _ccs[0] = numpy.floor(_ndx / 2) + 1
        _ccs[1] = numpy.floor(_ndy / 2) + 1
        if _center_channel[1] <= 1:
            _center_channel[1] = 1

    return _index_xs.astype(int) - 1, \
           _index_ys.astype(int) - 1, \
           _index_x0.astype(int) - 1, \
           _index_y0.astype(int) - 1, \
           _ccs.astype(int) - 1
