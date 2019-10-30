"""
prop_Control.py
"""
import abc
from dataclasses import dataclass

import numpy

from consts import ProfileHistory, NoAberrationAndHomogeneousMedium, AberrationFromFile, AberrationPhantom
from material.ab_phantom import AbPhantom
from material.muscle import Muscle
from misc.log2round_off import log2round_off


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
    abamp = 0.0
    ablength = 0.0
    abseed = 0.0

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

    @staticmethod
    def init_prop_control(simulation_name='beamsim',
                          num_dimensions=2,
                          config: Config = Config(
                              diffraction_type=ExactDiffraction,
                              non_linearity=True,
                              attenuation=True,
                              heterogeneous_medium=NoAberrationAndHomogeneousMedium,
                              annular_transducer=False,
                              equidistant_steps=False,
                              history=ProfileHistory),
                          harm=2,
                          fi=3,
                          bandwidth=0.5,
                          np=0,
                          amp=0.5,
                          material=Muscle(37.0),
                          temperature=37.0,
                          endpoint=0.1,
                          faz=0.06,
                          fel=[],
                          nelaz=None,
                          dimelaz=None,
                          nelel=None,
                          dimelel=None):
        if np == 0:
            np = 4 * numpy.sqrt(numpy.log(2)) / (numpy.pi * bandwidth)

        diffraction_type = config.diffraction_type
        non_linearity = config.non_linearity

        if config.annular_transducer:
            if config.heterogeneous_medium:
                diffraction_type = ExactDiffraction
                num_dimensions = 3
            if diffraction_type == NoDiffraction or \
                    diffraction_type == ExactDiffraction or \
                    diffraction_type == AngularSpectrumDiffraction:
                diffraction_type = PseudoDifferential
                num_dimensions = 2
            else:
                num_dimensions = 2
        if harm > 1:
            non_linearity = 1

        if nelaz is None:
            if config.annular_transducer:
                nelaz = 8
            else:
                nelaz = 64
        if dimelaz is None:
            if config.annular_transducer:
                dimelaz = 13e-4
            else:
                dimelaz = 3.5e-4
        if nelel is None:
            if config.annular_transducer:
                nelel = 8
            else:
                nelel = 1
        if dimelel is None:
            if config.annular_transducer:
                dimelel = 13e-4
            else:
                dimelel = 0.012

        # assign material struct
        if material is None:
            print('Material is not properly specified.')
            print(f'MUSCLE at {temperature} degrees is used')
            material = Muscle(temperature)

        # adjust frequency dependent variables
        fs = numpy.array([0.1, 0.5, 1.5, 3.0, 6.0, 12.0]) * 1e6
        ss = numpy.array([10, 5, 2.5, 1.25, 0.5, 0.25]) * 1e-3
        filter = numpy.array([1.0, 1.6, 2.0, 2.0, 2.0, 2.2, 2.2, 2.2, 2.2, 2.2]) / numpy.arange(1, 11) * 0.5
        fi = fi * 1e6
        ft = fi / harm
        c = material.sound_speed
        lambdai = c / fi
        if num_dimensions == 2:
            scale = 2
        else:
            scale = 1

        dx = lambdai / (2 * scale)
        ndxprel = numpy.ceil(dimelaz / dx)
        if numpy.mod(ndxprel, 2) == 0 and config.annular_transducer:
            ndxprel = ndxprel + 1
        dx = dimelaz / ndxprel
        if config.annular_transducer:
            daz = (2 * nelaz - 1) * dimelaz
        else:
            daz = nelaz * dimelaz

        dy = lambdai / (2 * scale)
        ndyprel = numpy.ceil(dimelel / dy)
        if numpy.mod(ndyprel, 2) == 0 and config.annular_transducer:
            ndyprel = ndyprel + 1
        dy = dimelel / ndyprel
        if config.annular_transducer:
            _del = (2 * nelel - 1) * dimelel
        else:
            _del = nelel * dimelel

        idx = numpy.where(abs(fi - fs) == min(abs(fi - fs)))[0][-1]
        stepsize = ss[idx]

        # calculate domain specific variables
        if not fel:
            fel = faz
        elif fel != (faz and config.annular_transducer):
            fel = faz

        if num_dimensions == 1:
            nlambdapad = 0
            nperiods = 12
        elif num_dimensions == 2:
            nlambdapad = 35
            nperiods = 12
        elif num_dimensions == 3:
            nlambdapad = 25
            nperiods = 8

        omegax = daz + 2 * nlambdapad * lambdai
        nptx = omegax / dx
        nx = log2round_off(nptx)
        if config.annular_transducer and \
                (diffraction_type == PseudoDifferential or
                 diffraction_type == FiniteDifferenceTimeDifferenceReduced or
                 diffraction_type == FiniteDifferenceTimeDifferenceFull):
            nx = nx / 2
        if num_dimensions == 3 and \
                (diffraction_type == NoDiffraction or
                 diffraction_type == ExactDiffraction or
                 diffraction_type == AngularSpectrumDiffraction):
            omegay = _del + 2 * nlambdapad * lambdai
            npty = omegay / dy
            ny = log2round_off(npty)
        else:
            ny = 1
        if num_dimensions == 1:
            nx = 1
            ny = 1

        Fs = numpy.maximum(40e6, 10.0 * ft)
        dt = 1.0 / Fs
        nptt = nperiods * (np / ft) / dt
        nt = log2round_off(nptt)

        new_config = Config(
            diffraction_type=diffraction_type,
            non_linearity=non_linearity,
            attenuation=config.attenuation,
            heterogeneous_medium=config.heterogeneous_medium,
            annular_transducer=config.annular_transducer,
            equidistant_steps=config.equidistant_steps,
            history=config.history)

        # simulation name
        prop_control = PropControl(simulation_name, num_dimensions, new_config)

        # domain and grid specifications
        prop_control.nx = nx
        prop_control.ny = ny
        prop_control.nt = nt
        prop_control.PMLwidth = 0

        # simulation parameters
        prop_control.stepsize = stepsize
        prop_control.nwindow = 2
        prop_control.shockstep = 0.5
        prop_control.endpoint = endpoint
        prop_control.currentpos = 0.0
        prop_control.storepos = [fel, faz]

        # material parameters
        prop_control.material = material
        if isinstance(material, AbPhantom):
            prop_control.d = 0.035
        else:
            prop_control.d = 0.02
        prop_control.offset = [0, 0]
        ns = 8
        prop_control.numscreens = ns
        prop_control.abamp = 0.09 * numpy.ones((ns, 1)) * 1e-3
        prop_control.ablength = numpy.ones((ns, 1)) * 1e-3 * numpy.array([4, 100])
        prop_control.abseed = numpy.arange(1, ns + 1)
        if config.heterogeneous_medium == AberrationFromFile:
            # TODO is it not a bool?
            prop_control.abfile = 'randseq.mat'
        elif config.heterogeneous_medium == AberrationPhantom:
            # TODO is it not a bool?
            prop_control.abfile = 'phantoml.mat'
        else:
            prop_control.abfile = ''

        # signal parameters
        prop_control.Fs = Fs
        prop_control.dx = dx
        prop_control.dy = dy
        prop_control.dz = c / (2.0 * numpy.pi * fi)
        prop_control.dt = dt
        prop_control.fc = ft
        prop_control.bandwidth = bandwidth * ft
        prop_control.np = np
        prop_control.amplitude = amp
        prop_control.harmonic = harm
        prop_control.filter = filter[0:harm]

        # transducer parameters
        prop_control.Dx = daz
        prop_control.Dy = _del
        prop_control.Fx = faz
        prop_control.Fy = fel
        if (new_config.diffraction_type == PseudoDifferential or
            new_config.diffraction_type == FiniteDifferenceTimeDifferenceReduced or
            new_config.diffraction_type == FiniteDifferenceTimeDifferenceFull) and new_config.annular_transducer:
            prop_control.cchannel = numpy.array([1, 1])
        else:
            prop_control.cchannel = numpy.floor([prop_control.nx / 2, prop_control.ny / 2]) + 1
        prop_control.cchannel.astype(int)
        prop_control.Nex = nelaz
        prop_control.Ney = nelel
        prop_control.esizex = dimelaz
        prop_control.esizey = dimelel

        if new_config.heterogeneous_medium:
            prop_control.storepos = numpy.array([prop_control.storepos, prop_control.d])
        prop_control.storepos = numpy.unique(prop_control.storepos)

        return prop_control
