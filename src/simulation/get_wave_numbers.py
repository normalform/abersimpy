# -*- coding: utf-8 -*-
"""
    get_wave_numbers.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
from typing import Optional

import numpy
from scipy.signal import hilbert

from simulation.controls.consts import SCALE_FOR_TEMPORAL_VARIABLE, SCALE_FOR_SPATIAL_VARIABLES_Z
from simulation.controls.main_control import MainControl
from simulation.filter.get_frequencies import get_frequencies
from system.diffraction.diffraction import NoDiffraction, ExactDiffraction, \
    AngularSpectrumDiffraction, PseudoDifferential, FiniteDifferenceTimeDifferenceFull, \
    FiniteDifferenceTimeDifferenceReduced


def get_wave_numbers(control: MainControl,
                     equidistant_steps: bool,
                     wave_number_operator: Optional[bool] = False):
    """
    Define wave number arrays in Fourier domain used for linear propagation and diffraction
    using the Angular Spectrum method.
    :param control:
    :param equidistant_steps: The flag specifying beam simulation with equidistant steps.
    :param wave_number_operator: Specifies the wave number operator for the classical Angular
        Spectrum Method of Zemp and Cobbold in
        regular coordinates (as opposed to retarded time coordinates).
    :return: Full complex wave number operator.
        If control.diffraction_type is set to PseudoDifferential, the wave numbers operator contains
        three layers, the first is the wave numbers in time and eigenvalues of difference matrix A.
        The second layer is the inverse eigenvector matrix Q, and the third the matrix Q.
    """
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t
    resolution_t = control.signal.resolution_t
    material = control.material.material
    sound_speed = material.sound_speed

    ft = 1 / resolution_t
    df = ft / num_points_t
    kt = 2.0 * numpy.pi / sound_speed * numpy.arange(-ft / 2,
                                                     ft / 2,
                                                     df)

    if control.diffraction_type in (NoDiffraction,
                                    ExactDiffraction,
                                    AngularSpectrumDiffraction):
        fx = 1 / control.signal.resolution_x
        fy = 1 / control.signal.resolution_y
        dkx = fx / num_points_x
        dky = fy / num_points_y
        if num_points_x == 1:
            kx = 0
        else:
            kx = 2.0 * numpy.pi * numpy.arange(-fx / 2, fx / 2, dkx)
        if num_points_y == 1:
            ky = 0
        else:
            ky = 2.0 * numpy.pi * numpy.arange(-fy / 2, fy / 2, dky)
    elif control.diffraction_type is PseudoDifferential:
        raise NotImplementedError
    elif control.diffraction_type in (FiniteDifferenceTimeDifferenceReduced,
                                      FiniteDifferenceTimeDifferenceFull):
        raise NotImplementedError

    # calculate attenuation if propagation is linear
    loss = numpy.zeros(kt.size)
    if control.attenuation and control.non_linearity is False:
        w = get_frequencies(num_points_t, control.signal.resolution_t / (
                2.0 * numpy.pi * SCALE_FOR_TEMPORAL_VARIABLE))
        eps_a = material.eps_a
        eps_b = material.eps_b
        loss = eps_a * numpy.conj(hilbert(numpy.abs(w) ** eps_b)) / SCALE_FOR_SPATIAL_VARIABLES_Z

    # assembly of wave-number operator
    if control.diffraction_type is AngularSpectrumDiffraction:
        # assign to vectors
        wave_numbers = numpy.zeros((numpy.max((num_points_x, num_points_y, num_points_t)), 4))
        wave_numbers[:num_points_x, 0] = numpy.fft.ifftshift(kx)
        wave_numbers[:num_points_y, 1] = numpy.fft.ifftshift(ky)
        wave_numbers[:num_points_t, 2] = numpy.fft.ifftshift(kt)
        wave_numbers[:num_points_t, 3] = loss
        return wave_numbers
    else:
        # building full complex wave number operator
        if control.num_dimensions == 3:
            ky2, kx2 = numpy.meshgrid(numpy.fft.ifftshift(ky ** 2),
                                      numpy.fft.ifftshift(kx ** 2),
                                      indexing='ij')
            kxy2 = kx2 + ky2
            kxy2 = kxy2.reshape((num_points_x * num_points_y, 1))
        elif control.num_dimensions == 2:
            kxy2 = numpy.fft.ifftshift(kx ** 2)
        else:
            kxy2 = 0
    kt, kxy = numpy.meshgrid(numpy.fft.ifftshift(kt), kxy2, indexing='ij')
    wave_numbers = numpy.sqrt((kt ** 2 - kxy).astype(complex))
    wave_numbers = numpy.sign(kt) * wave_numbers.real - 1j * wave_numbers.imag

    # introduces retarded time
    if wave_number_operator is False:
        wave_numbers = wave_numbers - kt

    # introduces loss in wave number operator
    if control.attenuation:
        for index in range(0, num_points_t):
            wave_numbers[index, ...] = wave_numbers[index, ...] - 1j * loss[index]

    # convert wave number operator to propagation operator
    if equidistant_steps:
        step_size = control.simulation.step_size
        if control.non_linearity:
            num_sub_steps = int(numpy.ceil(step_size / control.signal.resolution_z))
            step_size = step_size / num_sub_steps
        wave_numbers = numpy.exp(-1j * wave_numbers * step_size)

    if control.diffraction_type is PseudoDifferential:
        raise NotImplementedError

    return wave_numbers
