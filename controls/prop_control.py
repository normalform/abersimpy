"""
prop_Control.py
"""
import numpy

from controls import consts
from diffraction.diffraction import NoDiffraction, AngularSpectrumDiffraction, ExactDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull
from diffraction.interfaces import IDiffractionType
from material.aberration_phantom import AberrationPhantom
from material.interfaces import IMaterial
from material.muscle import Muscle
from misc.log2round_off import log2round_off


class Config:
    """
    Config
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 diffraction_type: IDiffractionType,
                 non_linearity: bool,
                 attenuation: bool,
                 heterogeneous_medium: int,
                 annular_transducer: bool,
                 equidistant_steps: bool,
                 history):
        self.diffraction_type = diffraction_type
        self.non_linearity = non_linearity
        self.attenuation = attenuation
        self.heterogeneous_medium = heterogeneous_medium
        self.annular_transducer = annular_transducer
        self.equidistant_steps = equidistant_steps
        self.history = history

    def __str__(self):
        msg = 'Config('
        msg += f'diffraction type:{self.diffraction_type}, '
        msg += f'non-linearity:{self.non_linearity}, '
        msg += f'attenuation:{self.attenuation}, '
        msg += f'heterogeneous_medium:{self.heterogeneous_medium}, '
        msg += f'annular transducer:{self.annular_transducer}, '
        msg += f'equidistant steps:{self.equidistant_steps}, '
        msg += f'history:{self.history})'

        return msg


class PropControl:
    """
    PropControl
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 simulation_name: str,
                 num_dimensions: int,
                 config: Config,
                 harmonic: int = 2):
        self.simulation_name = simulation_name
        self.config = config
        self.harmonic = harmonic

        self._init_prop_control(num_dimensions)

    def _init_prop_control(self,
                           num_dimensions: int,
                           image_frequency: float = 3.0e6,
                           band_width: float = 0.5,
                           num_periods: float = 0.0,
                           pulse_amplitude: float = 0.5,
                           material: IMaterial = Muscle(37.0),
                           end_point: float = 0.1,
                           focus_azimuth: float = 0.06,
                           focus_elevation: float = 0.06,
                           num_elements_azimuth: int = 64,
                           elements_size_azimuth: float = 3.5e-4,
                           num_elements_elevation: int = 1,
                           elements_size_elevation: float = 0.012):
        if num_periods == 0.0:
            num_periods = 4.0 * numpy.sqrt(numpy.log(2)) / (numpy.pi * band_width)

        _diffraction_type = self.config.diffraction_type
        _non_linearity = self.config.non_linearity

        if self.config.annular_transducer:
            if self.config.heterogeneous_medium:
                _diffraction_type = ExactDiffraction
                num_dimensions = 3
            elif _diffraction_type in (NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction):
                _diffraction_type = PseudoDifferential
                num_dimensions = 2
            else:
                num_dimensions = 2
        self.num_dimensions = num_dimensions

        if self.harmonic > 1:
            _non_linearity = 1

        # adjust frequency dependent variables
        _frequency_steps = numpy.array([0.1, 0.5, 1.5, 3.0, 6.0, 12.0]) * 1e6
        _step_sizes = numpy.array([10, 5, 2.5, 1.25, 0.5, 0.25]) * 1e-3
        _filters = numpy.array([1.0, 1.6, 2.0, 2.0, 2.0, 2.2, 2.2, 2.2, 2.2, 2.2]) / numpy.arange(1, 11) * 0.5
        _transmit_frequency = image_frequency / self.harmonic
        _sound_speed = material.sound_speed
        _lambda = _sound_speed / image_frequency
        if self.num_dimensions == 2:
            _scale = 2.0
        else:
            _scale = 1.0

        _resolution_x = _lambda / (2.0 * _scale)
        _ndx_preliminary = numpy.ceil(elements_size_azimuth / _resolution_x)
        if numpy.mod(_ndx_preliminary, 2) == 0 and self.config.annular_transducer:
            _ndx_preliminary = _ndx_preliminary + 1
        _resolution_x = elements_size_azimuth / _ndx_preliminary
        if self.config.annular_transducer:
            _probe_span_azimuth = (2 * num_elements_azimuth - 1) * elements_size_azimuth
        else:
            _probe_span_azimuth = num_elements_azimuth * elements_size_azimuth

        _resolution_y = _lambda / (2.0 * _scale)
        _ndy_preliminary = numpy.ceil(elements_size_elevation / _resolution_y)
        if numpy.mod(_ndy_preliminary, 2) == 0 and self.config.annular_transducer:
            _ndy_preliminary = _ndy_preliminary + 1
        _resolution_y = elements_size_elevation / _ndy_preliminary
        if self.config.annular_transducer:
            _probe_span_elevation = (2 * num_elements_elevation - 1) * elements_size_elevation
        else:
            _probe_span_elevation = num_elements_elevation * elements_size_elevation

        _idx = numpy.where(numpy.abs(image_frequency - _frequency_steps) ==
                           numpy.min(numpy.abs(image_frequency - _frequency_steps)))[0][-1]
        _step_size = _step_sizes[_idx]

        # calculate domain specific variables
        if self.config.annular_transducer:
            focus_elevation = focus_azimuth

        if self.num_dimensions == 1:
            _num_lambda_pad = 0
            _num_periods = 12
        elif self.num_dimensions == 2:
            _num_lambda_pad = 35
            _num_periods = 12
        elif self.num_dimensions == 3:
            _num_lambda_pad = 25
            _num_periods = 8
        else:
            raise ValueError(f'Unknown dimensions: {self.num_dimensions}')

        _omega_x = _probe_span_azimuth + 2 * _num_lambda_pad * _lambda
        _num_points_x = log2round_off(_omega_x / _resolution_x)
        if self.config.annular_transducer and _diffraction_type in (PseudoDifferential,
                                                                    FiniteDifferenceTimeDifferenceReduced,
                                                                    FiniteDifferenceTimeDifferenceFull):
            _num_points_x = _num_points_x / 2
        if self.num_dimensions == 3 and _diffraction_type in (NoDiffraction,
                                                              ExactDiffraction,
                                                              AngularSpectrumDiffraction):
            _omega_y = _probe_span_elevation + 2 * _num_lambda_pad * _lambda
            _num_points_y = log2round_off(_omega_y / _resolution_y)
        else:
            _num_points_y = 1
        if self.num_dimensions == 1:
            _num_points_x = 1
            _num_points_y = 1

        _sample_frequency = numpy.maximum(40e6, 10.0 * _transmit_frequency)
        _resolution_t = 1.0 / _sample_frequency
        _num_points_t = log2round_off(_num_periods * (num_periods / _transmit_frequency) / _resolution_t)

        new_config = Config(
            diffraction_type=_diffraction_type,
            non_linearity=_non_linearity,
            attenuation=self.config.attenuation,
            heterogeneous_medium=self.config.heterogeneous_medium,
            annular_transducer=self.config.annular_transducer,
            equidistant_steps=self.config.equidistant_steps,
            history=self.config.history)

        # domain and grid specifications
        self.num_points_x = _num_points_x
        self.num_points_y = _num_points_y
        self.num_points_t = _num_points_t
        self.perfect_matching_layer_width: float = 0.0

        # simulation parameters
        self.step_size = _step_size
        self.num_windows: int = 2
        self.shock_step: float = 0.5
        self.endpoint = end_point
        self.current_position: float = 0.0
        self.store_position = [focus_elevation, focus_azimuth]

        # material parameters
        self.material = material
        if isinstance(material, AberrationPhantom):
            self.thickness = 0.035
        else:
            self.thickness = 0.02
        self.offset = [0, 0]
        _num_screens = 8
        self.num_screens = _num_screens
        self.delay_screens_amplitude = 0.09 * numpy.ones((_num_screens, 1)) * 1e-3
        self.delay_screens_length = numpy.ones((_num_screens, 1)) * 1e-3 * numpy.array([4, 100])
        self.delay_screens_seed = numpy.arange(1, _num_screens + 1)
        if new_config.heterogeneous_medium == consts.AberrationFromFile:
            # TODO
            self.delay_screens_file = 'randseq.mat'
        elif new_config.heterogeneous_medium == consts.AberrationPhantom:
            # TODO
            self.delay_screens_file = 'phantoml.mat'
        else:
            self.delay_screens_file = ''

        # signal parameters
        self.sample_frequency = _sample_frequency
        self.resolution_x = _resolution_x
        self.resolution_y = _resolution_y
        self.resolution_z = _sound_speed / (2.0 * numpy.pi * image_frequency)
        self.resolution_t = _resolution_t
        self.transmit_frequency = _transmit_frequency
        self.bandwidth = band_width * _transmit_frequency
        self.num_periods = num_periods
        self.amplitude = pulse_amplitude
        self.harmonic = self.harmonic
        self.filter = _filters[0:self.harmonic]

        # transducer parameters
        self.probe_span_azimuth = _probe_span_azimuth
        self.probe_span_elevation = _probe_span_elevation
        self.focus_azimuth = focus_azimuth
        self.focus_elevation = focus_elevation
        if new_config.annular_transducer and new_config.diffraction_type in (PseudoDifferential,
                                                                             FiniteDifferenceTimeDifferenceReduced,
                                                                             FiniteDifferenceTimeDifferenceFull):
            self.center_channel = numpy.array([1, 1])
        else:
            self.center_channel = numpy.floor([self.num_points_x / 2, self.num_points_y / 2]) + 1
        self.center_channel.astype(int)
        self.num_elements_azimuth = num_elements_azimuth
        self.num_elements_elevation = num_elements_elevation
        self.elements_size_azimuth = elements_size_azimuth
        self.elements_size_elevation = elements_size_elevation

        if new_config.heterogeneous_medium:
            self.store_position = numpy.array([self.store_position, self.thickness])
        self.store_position = numpy.unique(self.store_position)
