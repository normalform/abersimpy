"""
PropControl
"""
from abersim_size import NS

import abc
from dataclasses import dataclass


class IDiffractionType(abc.ABC):
    """
    DiffractionType
    """


class NoDiffraction(IDiffractionType):
    """
    No diffraction
    """


class ExactDiffraction(IDiffractionType):
    """
    Exact diffraction using angular spectrum with Kz as a variable
    """


class AngularSpectrumDiffraction(IDiffractionType):
    """
    Angular spectrum with Kz as vectors (saves memory)
    """


class PseudoDifferential(IDiffractionType):
    """
    Pseudo differential model using matrix diagonalization for decoupling of equations
    """


class FiniteDifferenceTimeDifferenceReduced(IDiffractionType):
    """
    Using a finite difference time
    difference scheme in time and space with the parabolic
    approximation. The matrices used for differentiation
    are banded to improve computation time.
    """


class FiniteDifferenceTimeDifferenceFull(IDiffractionType):
    """
    Using a finite difference time
    difference scheme in time and space with the parabolic
    approximation. The matrices used for differentiation
    are full and requires more computation time.
    """


@dataclass
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


@dataclass
class PropControl:
    """
    PropControl
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 simulation_name: str,
                 num_dimensions: int,
                 config: Config):
        self.simulation_name = simulation_name
        self.num_dimensions = num_dimensions
        self.config = config

    nx = 0
    ny = 0
    nt = 0

    PMLwidth = 0

    stepsize = 0.0
    nwindow = 0.0
    shockstep = 0.0
    endpoint = 0.0
    currentpos = 0.0
    nstorepos = 0
    storepos = 0.0

    material = None

    d = 0.0
    offset = [0.0] * 2
    numscreens = 1
    abamp = [0.0] * NS
    ablength = [0.0] * NS * 2
    abseed = [0] * NS

    abfile = ''

    Fs = 0
    dx = 0.0
    dy = 0.0
    dz = 0.0
    dt = 0.0
    fc = 0.0
    bandwidth = 0.0
    Np = 0.0
    amplitude = 0.0
    harmonic = 0
    filter = [0.0] * 10

    Dx = 0.0
    Dy = 0.0
    Fx = 0.0
    Fy = 0.0
    cchannel = [0] * 2
    Nex = 0
    Ney = 0
    esizex = 0.0
    esizey = 0.0