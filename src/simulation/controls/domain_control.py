# -*- coding: utf-8 -*-
"""
    domain_control.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import numpy

from simulation.controls.log2_round_off import log2_round_off
from system.diffraction.diffraction import NoDiffraction, ExactDiffraction, \
    AngularSpectrumDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from system.diffraction.interfaces import IDiffractionType
from system.material.interfaces import IMaterial


class DomainControl:
    """
    DomainControl
    TODO Need unit tests
    """

    def __init__(self,
                 annular_transducer: bool,
                 num_dimensions: int,
                 harmonic: int,
                 material: IMaterial,
                 num_elements_azimuth: int,
                 elements_size_azimuth: float,
                 num_elements_elevation: int,
                 elements_size_elevation: float,
                 image_frequency: float,
                 num_periods: float,
                 diffraction_type: IDiffractionType):
        # adjust frequency dependent variables
        frequency_steps = numpy.array([0.1, 0.5, 1.5, 3.0, 6.0, 12.0]) * 1e6
        step_sizes = numpy.array([10, 5, 2.5, 1.25, 0.5, 0.25]) * 1e-3
        filters = numpy.array(
            [1.0, 1.6, 2.0, 2.0, 2.0, 2.2, 2.2, 2.2, 2.2, 2.2]) / numpy.arange(1, 11) * 0.5
        transmit_frequency = image_frequency / harmonic
        sound_speed = material.sound_speed
        _lambda = sound_speed / image_frequency
        if num_dimensions == 2:
            scale = 2.0
        else:
            scale = 1.0

        resolution_x = _lambda / (2.0 * scale)
        ndx_preliminary = numpy.ceil(elements_size_azimuth / resolution_x)
        if numpy.mod(ndx_preliminary, 2) == 0 and annular_transducer:
            ndx_preliminary = ndx_preliminary + 1
        resolution_x = elements_size_azimuth / ndx_preliminary
        if annular_transducer:
            probe_span_azimuth = (2 * num_elements_azimuth - 1) * elements_size_azimuth
        else:
            probe_span_azimuth = num_elements_azimuth * elements_size_azimuth

        resolution_y = _lambda / (2.0 * scale)
        ndy_preliminary = numpy.ceil(elements_size_elevation / resolution_y)
        if numpy.mod(ndy_preliminary, 2) == 0 and annular_transducer:
            ndy_preliminary = ndy_preliminary + 1
        resolution_y = elements_size_elevation / ndy_preliminary
        if annular_transducer:
            probe_span_elevation = (2 * num_elements_elevation - 1) * elements_size_elevation
        else:
            probe_span_elevation = num_elements_elevation * elements_size_elevation

        idx = numpy.where(numpy.abs(image_frequency - frequency_steps) ==
                          numpy.min(numpy.abs(image_frequency - frequency_steps)))[0][-1]
        step_size = step_sizes[idx]

        sample_frequency = numpy.maximum(40e6, 10.0 * transmit_frequency)
        resolution_t = 1.0 / sample_frequency

        # calculate domain specific variables
        if num_dimensions == 1:
            num_lambda_pad = 0
            _num_periods = 12
        elif num_dimensions == 2:
            num_lambda_pad = 35
            _num_periods = 12
        elif num_dimensions == 3:
            num_lambda_pad = 25
            _num_periods = 8
        else:
            raise ValueError(f'Unknown dimensions: {num_dimensions}')

        omega_x = probe_span_azimuth + 2 * num_lambda_pad * _lambda
        num_points_x = log2_round_off(omega_x / resolution_x)
        if annular_transducer and diffraction_type in (PseudoDifferential,
                                                       FiniteDifferenceTimeDifferenceReduced,
                                                       FiniteDifferenceTimeDifferenceFull):
            num_points_x = num_points_x / 2
        if num_dimensions == 3 and diffraction_type in (NoDiffraction,
                                                        ExactDiffraction,
                                                        AngularSpectrumDiffraction):
            omega_y = probe_span_elevation + 2 * num_lambda_pad * _lambda
            num_points_y = log2_round_off(omega_y / resolution_y)
        else:
            num_points_y = 1
        if num_dimensions == 1:
            num_points_x = 1
            num_points_y = 1

        num_points_t = log2_round_off(
            _num_periods * (num_periods / transmit_frequency) / resolution_t)

        # domain and grid specifications
        self._step_size = step_size
        self._sample_frequency = sample_frequency
        self._resolution_x = resolution_x
        self._resolution_y = resolution_y
        self._resolution_t = resolution_t
        self._sound_speed = sound_speed
        self._transmit_frequency = transmit_frequency
        self._filters = filters
        self._probe_span_azimuth = probe_span_azimuth
        self._probe_span_elevation = probe_span_elevation

        self._num_points_x = num_points_x
        self._num_points_y = num_points_y
        self._num_points_t = num_points_t
        self._perfect_matching_layer_width: float = 0.0

    @property
    def step_size(self) -> float:
        return self._step_size

    @property
    def sample_frequency(self) -> float:
        return self._sample_frequency

    @property
    def resolution_x(self) -> float:
        return self._resolution_x

    @property
    def resolution_y(self) -> float:
        return self._resolution_y

    @property
    def resolution_t(self) -> float:
        return self._resolution_t

    @property
    def sound_speed(self) -> float:
        return self._sound_speed

    @property
    def transmit_frequency(self) -> float:
        return self._transmit_frequency

    @property
    def filters(self) -> numpy.ndarray:
        return self._filters

    @property
    def probe_span_azimuth(self) -> float:
        return self._probe_span_azimuth

    @property
    def probe_span_elevation(self) -> float:
        return self._probe_span_elevation

    @property
    def num_points_x(self) -> int:
        return self._num_points_x

    @property
    def num_points_y(self) -> int:
        return self._num_points_y

    @property
    def num_points_t(self) -> int:
        return self._num_points_t

    @property
    def perfect_matching_layer_width(self) -> float:
        return self._perfect_matching_layer_width
