"""
get_wave_numbers.py

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy
from scipy.signal import hilbert

from simulation.controls.consts import ScaleForTemporalVariable, ScaleForSpatialVariablesZ
from simulation.controls.main_control import MainControl
from simulation.filter.get_frequencies import get_frequencies
from system.diffraction.diffraction import NoDiffraction, ExactDiffraction, \
    AngularSpectrumDiffraction, PseudoDifferential, FiniteDifferenceTimeDifferenceFull, \
    FiniteDifferenceTimeDifferenceReduced


def get_wave_numbers(control: MainControl,
                     equidistant_steps: bool,
                     wave_number_operator: bool = False):
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
    _num_points_x = control.domain.num_points_x
    _num_points_y = control.domain.num_points_y
    _num_points_t = control.domain.num_points_t
    _resolution_t = control.signal.resolution_t
    _material = control.material.material
    _sound_speed = _material.sound_speed

    _ft = 1 / _resolution_t
    _df = _ft / _num_points_t
    _kt = 2.0 * numpy.pi / _sound_speed * numpy.arange(-_ft / 2,
                                                       _ft / 2,
                                                       _df)

    if control.diffraction_type in (NoDiffraction,
                                    ExactDiffraction,
                                    AngularSpectrumDiffraction):
        _fx = 1 / control.signal.resolution_x
        _fy = 1 / control.signal.resolution_y
        _dkx = _fx / _num_points_x
        _dky = _fy / _num_points_y
        if _num_points_x is 1:
            _kx = 0
        else:
            _kx = 2.0 * numpy.pi * numpy.arange(-_fx / 2, _fx / 2, _dkx)
        if _num_points_y is 1:
            _ky = 0
        else:
            _ky = 2.0 * numpy.pi * numpy.arange(-_fy / 2, _fy / 2, _dky)
    elif control.diffraction_type is PseudoDifferential:
        raise NotImplementedError
    elif control.diffraction_type in (FiniteDifferenceTimeDifferenceReduced,
                                      FiniteDifferenceTimeDifferenceFull):
        raise NotImplementedError

    # calculate attenuation if propagation is linear
    _loss = numpy.zeros(_kt.size)
    if control.attenuation and control.non_linearity is False:
        _w = get_frequencies(_num_points_t, control.signal.resolution_t / (
                2.0 * numpy.pi * ScaleForTemporalVariable))
        _eps_a = _material.eps_a
        _eps_b = _material.eps_b
        _loss = _eps_a * numpy.conj(hilbert(numpy.abs(_w) ** _eps_b)) / ScaleForSpatialVariablesZ

    # assembly of wave-number operator
    if control.diffraction_type is AngularSpectrumDiffraction:
        # assign to vectors
        _wave_numbers = numpy.zeros((numpy.max((_num_points_x, _num_points_y, _num_points_t)), 4))
        _wave_numbers[:_num_points_x, 0] = numpy.fft.ifftshift(_kx)
        _wave_numbers[:_num_points_y, 1] = numpy.fft.ifftshift(_ky)
        _wave_numbers[:_num_points_t, 2] = numpy.fft.ifftshift(_kt)
        _wave_numbers[:_num_points_t, 3] = _loss
        return _wave_numbers
    else:
        # building full complex wave number operator
        if control.num_dimensions is 3:
            _ky2, _kx2 = numpy.meshgrid(numpy.fft.ifftshift(_ky ** 2),
                                        numpy.fft.ifftshift(_kx ** 2),
                                        indexing='ij')
            _kxy2 = _kx2 + _ky2
            _kxy2 = _kxy2.reshape((_num_points_x * _num_points_y, 1))
        elif control.num_dimensions is 2:
            _kxy2 = numpy.fft.ifftshift(_kx ** 2)
        else:
            _kxy2 = 0
    _kt, _kxy = numpy.meshgrid(numpy.fft.ifftshift(_kt), _kxy2, indexing='ij')
    _wave_numbers = numpy.sqrt((_kt ** 2 - _kxy).astype(complex))
    _wave_numbers = numpy.sign(_kt) * _wave_numbers.real - 1j * _wave_numbers.imag

    # introduces retarded time
    if wave_number_operator is False:
        _wave_numbers = _wave_numbers - _kt

    # introduces _loss in wave number operator
    if control.attenuation:
        for _index in range(0, _num_points_t):
            _wave_numbers[_index, ...] = _wave_numbers[_index, ...] - 1j * _loss[_index]

    # convert wave number operator to propagation operator
    if equidistant_steps:
        _step_size = control.simulation.step_size
        if control.non_linearity:
            _num_sub_steps = int(numpy.ceil(_step_size / control.signal.resolution_z))
            _step_size = _step_size / _num_sub_steps
        _wave_numbers = numpy.exp(-1j * _wave_numbers * _step_size)

    if control.diffraction_type is PseudoDifferential:
        raise NotImplementedError

    return _wave_numbers
