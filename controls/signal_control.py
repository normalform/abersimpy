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
                 image_frequency: float,
                 bandwidth: float,
                 num_periods: float,
                 pulse_amplitude: float,
                 harmonic: int):
        # signal control parameters
        self._sample_frequency = domain.sample_frequency
        self._resolution_x = domain.resolution_x
        self._resolution_y = domain.resolution_y
        self._resolution_z = domain.sound_speed / (2.0 * numpy.pi * image_frequency)
        self._resolution_t = domain.resolution_t
        self._transmit_frequency = domain.transmit_frequency
        self._bandwidth = bandwidth * domain.transmit_frequency
        self._num_periods = num_periods
        self._amplitude = pulse_amplitude
        self._filter = domain.filters[0:harmonic]

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
    def resolution_z(self) -> float:
        return self._resolution_z

    @property
    def resolution_t(self) -> float:
        return self._resolution_t

    @property
    def transmit_frequency(self) -> float:
        return self._transmit_frequency

    @property
    def bandwidth(self) -> float:
        return self._bandwidth

    @property
    def num_periods(self) -> float:
        return self._num_periods

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @property
    def filter(self) -> numpy.ndarray:
        return self._filter
