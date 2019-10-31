"""
signal_control.py
"""
import numpy

from controls.domain_control import DomainControl


class SignalControl:
    """
    SignalControl
    """

    def __init__(self,
                 domain: DomainControl,
                 image_frequency,
                 bandwidth,
                 num_periods,
                 pulse_amplitude,
                 harmonic):
        # signal control parameters
        self.sample_frequency = domain.sample_frequency
        self.resolution_x = domain.resolution_x
        self.resolution_y = domain.resolution_y
        self.resolution_z = domain.sound_speed / (2.0 * numpy.pi * image_frequency)
        self.resolution_t = domain.resolution_t
        self.transmit_frequency = domain.transmit_frequency
        self.bandwidth = bandwidth * domain.transmit_frequency
        self.num_periods = num_periods
        self.amplitude = pulse_amplitude
        self.filter = domain.filters[0:harmonic]
