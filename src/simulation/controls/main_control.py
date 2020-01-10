# -*- coding: utf-8 -*-
"""
    main_control.py

    :copyright (C) 2020  Jaeho
    :license: GPL-3.0
"""
from typing import Type

import numpy

from simulation.controls.consts import PROFILE_HISTORY
from simulation.controls.domain_control import DomainControl
from simulation.controls.material_control import MaterialControl
from simulation.controls.signal_control import SignalControl
from simulation.controls.simulation_control import SimulationControl
from simulation.controls.transducer_control import TransducerControl
from system.diffraction.diffraction import ExactDiffraction, PseudoDifferential
from system.diffraction.diffraction import NoDiffraction, AngularSpectrumDiffraction
from system.diffraction.interfaces import IDiffractionType
from system.material.interfaces import IMaterial
from system.material.muscle import Muscle


class MainControl:
    """
    MainControl
    TODO Need unit tests
    TODO Might need better name.
    TODO Might need better structure with other sub controls.
    TODO Need Json serialization & deserialization
    """

    def __init__(self,
                 simulation_name: str,
                 num_dimensions: int,
                 diffraction_type: Type[IDiffractionType],
                 non_linearity: bool,
                 attenuation: bool,
                 heterogeneous_medium: int,
                 history: int = PROFILE_HISTORY,
                 annular_transducer: bool = False,
                 equidistant_steps: bool = False,
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
        """
        Constructor
        :param simulation_name: The simulation name.
        :param num_dimensions: The number of dimensions.
            A 3D simulation using a circular symmetric transducer may be
            utilized when specifying a 2D simulation together with a annular transducer and
            heterogeneous medium. If an annular transducer is specified together
            with a heterogeneous medium, the annular transducer will be formed in full 3D
            coordinates and num_dimensions automatically set to 3.
        :param diffraction_type: The type of diffraction.
            NoDiffraction. No diffraction
            ExactDiffraction. Exact diffraction using angular spectrum with wave number
                operator(Kz) as a variable.
            AngularSpectrumDiffraction. Angular spectrum with wave number operator(Kz)
                as vectors (saves memory).
            PseudoDifferential. Pseudo differential model using matrix
                diagonalization for decoupling of equations.
            FiniteDifferenceTimeDifferenceReduced. Using a finite difference time difference scheme
                in time and space with the parabolic approximation. The matrices used for
                differentiation are banded to improve computation time.
            FiniteDifferenceTimeDifferenceFull. Using a finite difference time difference scheme
                in time and space with the parabolic approximation.
                The matrices used for differentiation are full and requires more computation time.
        :param non_linearity: Linear or non-linear medium.
            False. Linear
            True. Non-linear
        :param attenuation: The frequency dependent power-law attenuation.
            False. No attenuation
            True. Attenuation on
        :param heterogeneous_medium: The heterogeneity type of medium.
        :param history: Simulation history policy.
            0. NOHISTORY. Store only pulse at endpoint.
            1. POSHISTORY. Store the pulse at the positions specified in simulation.store_position
            2. PROFHISTORY. Store temporal maximum and RMS profiles for each step and store the
                pulse at the positions specified in simulation.store_position.
                If only the profiles are wanted, set store_position to an empty vector.
            3. FULLHISTORY. Store whole pulse for each step.
            4. PLANEHISTORY. Store the azimuth and elevation plane of each pulse and the
                whole pulse at depths specified in simulation.store_position.
            5. AXPHISTORY. As PLANEHISTORY, but only the pulse specified by
                transducer.center_channel is exported.
        :param annular_transducer:
            If the transducer surface is circular, the annular_transducer should be set to True.
            This enables the use of a 2D computation domain if the medium is homogeneous.
            For arbitrary transducer geometries annular_transducer must be set to False.
        :param equidistant_steps: The flag specifying beam simulation with equidistant steps.
            Speeds up computation, but should be used with care
                False. Variable step size
                True. Constant step size
        :param harmonic: The harmonic frequency used for imaging, i.e., if harmonic is set to 2,
            then second-harmonic imaging is performed.
            If harmonic is set to 3, then third harmonic and so forth.
        :param image_frequency: Imaging frequency in MHz
        :param band_width: Relative bandwidth of transmitted pulse.
            This is not the bandwidth of the transducer.
        :param num_periods: Length of Gaussian pulse in periods.
        :param pulse_amplitude: Amplitude of pulse in MPa. If a MI-limited amplitude is desirable,
            please see transducer.estimate_amplitude for more information.
        :param material: The type of material.
        :param end_point: The end point of simulation. (e.g., 0.1 is for 10cm)
        :param focus_azimuth: The focus in azimuth direction.
        :param focus_elevation: The focus in elevation direction.
        :param num_elements_azimuth: The number of elements in azimuth direction.
        :param elements_size_azimuth: The dimension of elements in azimuth direction.
            Default is 0.35 mm (~lambda/2 of 2.2 MHz and approx. of ~22 mm for 64 elements.).
            For annular transducers, the default element size is 1.3mm.
        :param num_elements_elevation: The number of elements in elevation direction.
        :param elements_size_elevation: The dimension of elements in elevation direction.
        """
        self._simulation_name = simulation_name
        self._harmonic = harmonic
        self._annular_transducer = annular_transducer
        self._heterogeneous_medium = heterogeneous_medium
        self._attenuation = attenuation
        self._equidistant_steps = equidistant_steps
        self._history = history

        if harmonic > 1:
            _non_linearity = True
        else:
            _non_linearity = non_linearity
        self._non_linearity = _non_linearity

        if num_periods == 0.0:
            num_periods = 4.0 * numpy.sqrt(numpy.log(2)) / (numpy.pi * band_width)

        if annular_transducer:
            if heterogeneous_medium:
                _diffraction_type = ExactDiffraction
                _num_dimensions = 3
            elif diffraction_type in (NoDiffraction, ExactDiffraction, AngularSpectrumDiffraction):
                _diffraction_type = PseudoDifferential
                _num_dimensions = 2
            else:
                _diffraction_type = diffraction_type
                _num_dimensions = 2
        else:
            _diffraction_type = diffraction_type
            _num_dimensions = num_dimensions
        self._diffraction_type = _diffraction_type
        self._num_dimensions = _num_dimensions

        # domain controls parameters
        self._domain = DomainControl(annular_transducer,
                                     self.num_dimensions,
                                     harmonic,
                                     material,
                                     num_elements_azimuth,
                                     elements_size_azimuth,
                                     num_elements_elevation,
                                     elements_size_elevation,
                                     image_frequency,
                                     num_periods,
                                     _diffraction_type)

        # material control parameters
        self._material = MaterialControl(material, heterogeneous_medium)

        # simulation control parameters
        self._simulation = SimulationControl(self._domain,
                                             end_point,
                                             focus_elevation,
                                             focus_azimuth,
                                             heterogeneous_medium,
                                             self._material)

        # signal control parameters
        self._signal = SignalControl(self._domain,
                                     image_frequency,
                                     band_width,
                                     num_periods,
                                     pulse_amplitude,
                                     harmonic)

        # transducer control parameters
        self._transducer = TransducerControl(self._domain,
                                             num_elements_azimuth,
                                             num_elements_elevation,
                                             elements_size_azimuth,
                                             elements_size_elevation,
                                             focus_azimuth,
                                             focus_elevation,
                                             _diffraction_type,
                                             self._domain.num_points_x,
                                             self._domain.num_points_y,
                                             annular_transducer)

    @property
    def simulation_name(self) -> str:
        """
        The simulation name
        :return: The simulation name
        """
        return self._simulation_name

    @property
    def num_dimensions(self) -> int:
        """
        The number of dimensions. A 3D simulation using a circular symmetric transducer may be
        utilized when specifying a 2D simulation together with a annular transducer and
        heterogeneous medium. If an annular transducer is specified together with a heterogeneous
        medium, the annular transducer will be formed in full 3D coordinates and num_dimensions
        automatically set to 3.
        :return: The number of dimensions.
        """
        return self._num_dimensions

    @property
    def harmonic(self) -> int:
        """
        The harmonic frequency used for imaging, i.e., if harmonic is set to 2, then second-harmonic
        imaging is performed. If harmonic is set to 3, then third harmonic and so forth.
        :return: The harmonic.
        """
        return self._harmonic

    @property
    def annular_transducer(self) -> bool:
        """
        If the transducer surface is circular, the annular_transducer should be set to True.
        This enables the use of a 2D computation domain if the medium is homogeneous.
        For arbitrary transducer geometries annular_transducer must be set to False.
        :return: True if the transducer surface is circular.
        """
        return self._annular_transducer

    @property
    def diffraction_type(self) -> IDiffractionType:
        """
        The type of diffraction
            NoDiffraction. No diffraction
            ExactDiffraction. Exact diffraction using angular spectrum
                with wave number operator(Kz) as a variable
            AngularSpectrumDiffraction. Angular spectrum with wave number operator(Kz)
                as vectors (saves memory)
            PseudoDifferential. Pseudo differential model using matrix diagonalization for
                decoupling of equations
            FiniteDifferenceTimeDifferenceReduced. Using a finite difference time difference scheme
                in time and space with the parabolic approximation. The matrices used for
                differentiation are banded to improve computation time.
            FiniteDifferenceTimeDifferenceFull. Using a finite difference time difference scheme
                in time and space with the parabolic approximation.
                The matrices used for differentiation are full and requires more computation time.
        :return: The type of diffraction
        """
        return self._diffraction_type

    @property
    def heterogeneous_medium(self) -> int:
        """
        The heterogeneity type of medium
        :return: The heterogeneity type of medium
        """
        return self._heterogeneous_medium

    @property
    def attenuation(self) -> bool:
        """
        The frequency dependent power-law attenuation
            False. No attenuation
            True. Attenuation on
        :return: The frequency dependent power-law attenuation.
        """
        return self._attenuation

    @property
    def equidistant_steps(self) -> bool:
        """
        The flag specifying beam simulation with equidistant steps.
        Speeds up computation, but should be used with care
            False. Variable step size
            True. Constant step size
        :return: True for constant step size.
        """
        return self._equidistant_steps

    @property
    def history(self) -> int:
        """
        Simulation history policy.
            0. NOHISTORY. Store only pulse at endpoint.
            1. POSHISTORY. Store the pulse at the positions specified in simulation.store_position
            2. PROFHISTORY. Store temporal maximum and RMS profiles for each step and store
                the pulse at the positions specified in simulation.store_position.
                If only the profiles are wanted, set store_position to an empty vector.
            3. FULLHISTORY. Store whole pulse for each step.
            4. PLANEHISTORY. Store the azimuth and elevation plane of each pulse and the
                whole pulse at depths specified in simulation.store_position.
            5. AXPHISTORY. As PLANEHISTORY, but only the pulse specified by
                transducer.center_channel is exported.
        :return: The history type.
        """
        return self._history

    @property
    def non_linearity(self) -> bool:
        """
        Linear or non-linear medium
            False. Linear
            True. Non-linear
        :return: The non-linearity
        """
        return self._non_linearity

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
