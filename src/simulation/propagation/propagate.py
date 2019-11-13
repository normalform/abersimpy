"""
propagate.py
"""
from typing import Optional

import numpy

from simulation.controls.main_control import MainControl
from simulation.get_wave_numbers import get_wave_numbers
from simulation.propagation.nonlinear.nonlinear_propagate import nonlinear_propagate
from system.diffraction.diffraction import ExactDiffraction, AngularSpectrumDiffraction, \
    PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull


def propagate(control: MainControl,
              wave: numpy.ndarray,
              direction: int,
              equidistant_steps: bool,
              wave_numbers: Optional[numpy.ndarray] = None) -> numpy.ndarray:
    """
    Function that handles propagation of 3D wave field in z-direction using the method of
    angular spectrum. The function will forward the handling of propagation to
    nonlinear_propagate if control.non_linearity is set to True.
    :param control: The controls.
    :param wave: Wave at position z: wave(x,y,z,t)
    :param direction: Direction of propagation.
        1 - positive z-direction
        -1 - negative z-direction
        For the values +/-2 linear propagation is chosen even if control.non_linearity is True.
    :param equidistant_steps: The flag specifying beam simulation with equidistant steps.
    :param wave_numbers: The wave numbers for the whole region.
    :return: The resulting field after propagation: wave(x,y,z+step_size,t)
    """
    if wave_numbers is None:
        _wave_numbers = get_wave_numbers(control, equidistant_steps)
    else:
        _wave_numbers = wave_numbers

    _diffraction_type = control.diffraction_type
    _non_linearity = control.non_linearity
    _attenuation = control.attenuation
    _step_size = control.simulation.step_size

    _num_points_x = control.domain.num_points_x
    _num_points_y = control.domain.num_points_y
    _num_points_t = control.domain.num_points_t

    # Update position and chose propagation mode
    if direction > 0:
        control.simulation.current_position = control.simulation.current_position + _step_size
    elif direction < 0:
        control.simulation.current_position = control.simulation.current_position - _step_size
    if (_diffraction_type == ExactDiffraction or
        _diffraction_type == AngularSpectrumDiffraction or
        _diffraction_type == PseudoDifferential) and \
            (_non_linearity is False or abs(direction) == 2):
        # Linear propagation and exact diffraction
        if _wave_numbers.size == 0:
            _wave_numbers = get_wave_numbers(control, equidistant_steps)

        # Forward spatial transform
        if _diffraction_type == ExactDiffraction or _diffraction_type == AngularSpectrumDiffraction:
            if control.num_dimensions == 3:
                _wave = numpy.fft.fftn(wave, axes=(2,))
                _wave = numpy.fft.fftn(wave, axes=(1,))
                _wave = _wave.reshape((_num_points_t, _num_points_x * _num_points_y))
            else:
                _wave = numpy.fft.fftn(wave, axes=(1,))
        elif _diffraction_type == PseudoDifferential:
            # TODO Need to complete
            tmp = _wave_numbers[:, _num_points_x:]
            raise NotImplementedError
        else:
            _wave = wave

        # Forward temporal FFT
        _wave = numpy.fft.fftn(_wave, axes=(0,))  # FFT in time

        # Propagation step
        if _diffraction_type in (ExactDiffraction,
                                 PseudoDifferential):
            if equidistant_steps:
                _wave = _wave * numpy.squeeze(
                    _wave_numbers[:_num_points_t, :_num_points_x * _num_points_y])
            else:
                _wave = _wave * numpy.exp(
                    (-1j * _step_size) * numpy.squeeze(
                        _wave_numbers[:_num_points_t, :_num_points_x * _num_points_y, 0]))
        elif _diffraction_type is AngularSpectrumDiffraction:
            raise NotImplementedError
            _kx = _wave_numbers[:_num_points_x, 0]
            _ky = _wave_numbers[:_num_points_y, 1]
            _kt = _wave_numbers[:_num_points_t, 2]
            _loss = _wave_numbers[:_num_points_t, 3]
            _index = 1
            for _index_x in range(_num_points_x):
                _kx2 = _kx[_index_x] ** 2
                for _index_y in range(_num_points_y):
                    ky2 = _ky[_index_y] ** 2
                    # Calculate propagation wave numbers
                    kz = numpy.sqrt(_kt ** 2 - _kx2 - ky2)
                    # Dampen evanescent waves
                    kz = numpy.sign(_kt) * kz.real - 1j * kz.imag
                    kz = kz - _kt - 1j * _loss
                    _wave[:, _index] = _wave[:, _index] * numpy.exp((-1j * _step_size) * kz)
                    _index = _index + 1

        # Backward temporal FFT
        _wave = numpy.fft.ifftn(_wave, axes=(0,))

        # Backward spatial transform
        if _diffraction_type in (ExactDiffraction,
                                 AngularSpectrumDiffraction):
            if control.num_dimensions == 3:
                _wave = _wave.reshape((_num_points_t, _num_points_y, _num_points_x))
                _wave = numpy.fft.ifftn(_wave, axes=(1,))
                _wave = numpy.fft.ifftn(_wave, axes=(2,))
            else:
                _wave = numpy.fft.ifftn(_wave, axes=(1,))
            _wave = _wave.real
        elif _diffraction_type is PseudoDifferential:
            _wave = _wave.real
            raise NotImplementedError
        elif _diffraction_type in (FiniteDifferenceTimeDifferenceReduced,
                                   FiniteDifferenceTimeDifferenceFull) or \
                (_non_linearity or _attenuation):
            # Nonlinear propagation in external function
            raise NotImplementedError
    elif _diffraction_type in (FiniteDifferenceTimeDifferenceReduced,
                               FiniteDifferenceTimeDifferenceFull) or \
            (_non_linearity or _attenuation):
        # Nonlinear propagation in external function
        _wave = nonlinear_propagate(control, wave, direction, equidistant_steps, _wave_numbers)
    else:
        print('Propagation type must be specified')
        exit(-1)

    return _wave
