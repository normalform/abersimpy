"""
transducer_control.py
"""
import numpy

from controls.domain_control import DomainControl
from diffraction.diffraction import PseudoDifferential, \
    FiniteDifferenceTimeDifferenceFull, FiniteDifferenceTimeDifferenceReduced


class TransducerControl:
    """
    TransducerControl
    """

    def __init__(self,
                 domain: DomainControl,
                 num_elements_azimuth,
                 num_elements_elevation,
                 elements_size_azimuth,
                 elements_size_elevation,
                 focus_azimuth: float,
                 focus_elevation: float,
                 diffraction_type,
                 num_points_x,
                 num_points_y,
                 annular_transducer: bool):
        # transducer control parameters
        self.num_elements_azimuth = num_elements_azimuth
        self.num_elements_elevation = num_elements_elevation
        self.elements_size_azimuth = elements_size_azimuth
        self.elements_size_elevation = elements_size_elevation
        self.probe_span_azimuth = domain.probe_span_azimuth
        self.probe_span_elevation = domain.probe_span_elevation
        self.focus_azimuth = focus_azimuth
        self.focus_elevation = focus_elevation
        if annular_transducer and diffraction_type in (PseudoDifferential,
                                                       FiniteDifferenceTimeDifferenceReduced,
                                                       FiniteDifferenceTimeDifferenceFull):
            self.center_channel = numpy.array([1, 1])
        else:
            self.center_channel = numpy.floor([num_points_x / 2, num_points_y / 2]) + 1
        self.center_channel.astype(int)
