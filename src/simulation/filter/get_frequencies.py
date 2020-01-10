# -*- coding: utf-8 -*-
"""
    get_frequencies.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
import numpy
from numpy.fft import ifftshift


def get_frequencies(num_frequency: int,
                    resolution_t: float) -> numpy.ndarray:
    """
    The function returns an array of physical, and not angular, frequencies of length num_frequency
    and with maximum frequency of 1/(2*resolution_t) (the Nyquist frequency)
    ranging from (-num_frequency/2:num_frequency/2-1)/resolution_t.
    TODO Need unit tests
    :param num_frequency: Length of frequency vector.
    :param resolution_t: Sampling interval.
    :return: Array of frequencies.
    """
    df = 1.0 / (resolution_t * num_frequency)
    frequencies = ifftshift(numpy.arange(0, num_frequency) - numpy.floor(num_frequency / 2)) * df

    return frequencies
