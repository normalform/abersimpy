"""
domain_control.py
"""
import numpy

from diffraction.diffraction import NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction, \
    PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from diffraction.interfaces import IDiffractionType
from material.interfaces import IMaterial
from misc.log2round_off import log2round_off


class DomainControl:
    """
    DomainControl
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
        _frequency_steps = numpy.array([0.1, 0.5, 1.5, 3.0, 6.0, 12.0]) * 1e6
        _step_sizes = numpy.array([10, 5, 2.5, 1.25, 0.5, 0.25]) * 1e-3
        _filters = numpy.array(
            [1.0, 1.6, 2.0, 2.0, 2.0, 2.2, 2.2, 2.2, 2.2, 2.2]) / numpy.arange(1, 11) * 0.5
        _transmit_frequency = image_frequency / harmonic
        _sound_speed = material.sound_speed
        _lambda = _sound_speed / image_frequency
        if num_dimensions == 2:
            _scale = 2.0
        else:
            _scale = 1.0

        _resolution_x = _lambda / (2.0 * _scale)
        _ndx_preliminary = numpy.ceil(elements_size_azimuth / _resolution_x)
        if numpy.mod(_ndx_preliminary, 2) == 0 and annular_transducer:
            _ndx_preliminary = _ndx_preliminary + 1
        _resolution_x = elements_size_azimuth / _ndx_preliminary
        if annular_transducer:
            _probe_span_azimuth = (2 * num_elements_azimuth - 1) * elements_size_azimuth
        else:
            _probe_span_azimuth = num_elements_azimuth * elements_size_azimuth

        _resolution_y = _lambda / (2.0 * _scale)
        _ndy_preliminary = numpy.ceil(elements_size_elevation / _resolution_y)
        if numpy.mod(_ndy_preliminary, 2) == 0 and annular_transducer:
            _ndy_preliminary = _ndy_preliminary + 1
        _resolution_y = elements_size_elevation / _ndy_preliminary
        if annular_transducer:
            _probe_span_elevation = (2 * num_elements_elevation - 1) * elements_size_elevation
        else:
            _probe_span_elevation = num_elements_elevation * elements_size_elevation

        _idx = numpy.where(numpy.abs(image_frequency - _frequency_steps) ==
                           numpy.min(numpy.abs(image_frequency - _frequency_steps)))[0][-1]
        _step_size = _step_sizes[_idx]

        _sample_frequency = numpy.maximum(40e6, 10.0 * _transmit_frequency)
        _resolution_t = 1.0 / _sample_frequency

        # calculate domain specific variables
        if num_dimensions == 1:
            _num_lambda_pad = 0
            _num_periods = 12
        elif num_dimensions == 2:
            _num_lambda_pad = 35
            _num_periods = 12
        elif num_dimensions == 3:
            _num_lambda_pad = 25
            _num_periods = 8
        else:
            raise ValueError(f'Unknown dimensions: {num_dimensions}')

        _omega_x = _probe_span_azimuth + 2 * _num_lambda_pad * _lambda
        _num_points_x = log2round_off(_omega_x / _resolution_x)
        if annular_transducer and diffraction_type in (PseudoDifferential,
                                                       FiniteDifferenceTimeDifferenceReduced,
                                                       FiniteDifferenceTimeDifferenceFull):
            _num_points_x = _num_points_x / 2
        if num_dimensions == 3 and diffraction_type in (NoDiffraction,
                                                        ExactDiffraction,
                                                        AngularSpectrumDiffraction):
            _omega_y = _probe_span_elevation + 2 * _num_lambda_pad * _lambda
            _num_points_y = log2round_off(_omega_y / _resolution_y)
        else:
            _num_points_y = 1
        if num_dimensions == 1:
            _num_points_x = 1
            _num_points_y = 1

        _num_points_t = log2round_off(
            _num_periods * (num_periods / _transmit_frequency) / _resolution_t)

        # domain and grid specifications
        self._step_size = _step_size
        self._sample_frequency = _sample_frequency
        self._resolution_x = _resolution_x
        self._resolution_y = _resolution_y
        self._resolution_t = _resolution_t
        self._sound_speed = _sound_speed
        self._transmit_frequency = _transmit_frequency
        self._filters = _filters
        self._probe_span_azimuth = _probe_span_azimuth
        self._probe_span_elevation = _probe_span_elevation

        self._num_points_x = _num_points_x
        self._num_points_y = _num_points_y
        self._num_points_t = _num_points_t
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
