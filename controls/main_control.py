"""
main_Control.py
"""
import numpy

from controls.config_control import ConfigControl
from controls.domain_control import DomainControl
from controls.material_control import MaterialControl
from controls.signal_control import SignalControl
from controls.simulation_control import SimulationControl
from controls.transducer_control import TransducerControl
from diffraction.diffraction import ExactDiffraction, PseudoDifferential
from diffraction.diffraction import NoDiffraction, AngularSpectrumDiffraction
from material.interfaces import IMaterial
from material.muscle import Muscle


class MainControl:
    """
    MainControl
    """

    def __init__(self,
                 simulation_name: str,
                 num_dimensions: int,
                 config: ConfigControl,
                 harmonic: int = 2,
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
        self.simulation_name = simulation_name
        self.config = config
        self.harmonic = harmonic

        if num_periods == 0.0:
            num_periods = 4.0 * numpy.sqrt(numpy.log(2)) / (numpy.pi * band_width)

        _diffraction_type = self.config.diffraction_type

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

        # domain controls parameters
        self._domain = DomainControl(self.config,
                                     self.num_dimensions,
                                     self.harmonic,
                                     material,
                                     num_elements_azimuth,
                                     elements_size_azimuth,
                                     num_elements_elevation,
                                     elements_size_elevation,
                                     image_frequency,
                                     num_periods,
                                     _diffraction_type)

        if self.harmonic > 1:
            _non_linearity = 1
        else:
            _non_linearity = self.config.non_linearity

        _new_config = ConfigControl(
            diffraction_type=_diffraction_type,
            non_linearity=_non_linearity,
            attenuation=self.config.attenuation,
            heterogeneous_medium=self.config.heterogeneous_medium,
            annular_transducer=self.config.annular_transducer,
            equidistant_steps=self.config.equidistant_steps,
            history=self.config.history)

        # material control parameters
        self._material = MaterialControl(material, _new_config)

        # simulation control parameters
        self._simulation = SimulationControl(self.domain,
                                             end_point,
                                             focus_elevation,
                                             focus_azimuth,
                                             _new_config,
                                             self.material)

        # signal control parameters
        self._signal = SignalControl(self.domain,
                                     image_frequency,
                                     band_width,
                                     num_periods,
                                     pulse_amplitude,
                                     self.harmonic)

        # transducer control parameters
        self._transducer = TransducerControl(self.domain,
                                             num_elements_azimuth,
                                             num_elements_elevation,
                                             elements_size_azimuth,
                                             elements_size_elevation,
                                             focus_azimuth,
                                             focus_elevation,
                                             _new_config.diffraction_type,
                                             self.domain.num_points_x,
                                             self.domain.num_points_y,
                                             _new_config.annular_transducer)

    @property
    def domain(self) -> DomainControl:
        """
        Get domain control
        :return: The domain control
        """
        return self._domain

    @property
    def material(self) -> MaterialControl:
        """
        Get material control
        :return: The material control
        """
        return self._material

    @property
    def simulation(self) -> SimulationControl:
        """
        Get simulation control
        :return: The simulation control
        """
        return self._simulation

    @property
    def signal(self) -> SignalControl:
        """
        Get signal control
        :return: The signal control
        """
        return self._signal

    @property
    def transducer(self) -> TransducerControl:
        """
        Get transducer control
        :return: The transducer control
        """
        return self._transducer
