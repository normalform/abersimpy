"""
init_prop_control.py
"""
import numpy

from consts import ProfileHistory, NoAberrationAndHomogeneousMedium, AberrationFromFile, AberrationPhantom
from material.ab_phantom import AbPhantom
from material.muscle import Muscle
from misc.log2round_off import log2round_off
from propcontrol import PropControl, Config, \
    NoDiffraction, AngularSpectrumDiffraction, ExactDiffraction, PseudoDifferential, \
    FiniteDifferenceTimeDifferenceReduced, FiniteDifferenceTimeDifferenceFull


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
    propcontrol = PropControl(simulation_name, num_dimensions, new_config)

    # domain and grid specifications
    propcontrol.nx = nx
    propcontrol.ny = ny
    propcontrol.nt = nt
    propcontrol.PMLwidth = 0

    # simulation parameters
    propcontrol.stepsize = stepsize
    propcontrol.nwindow = 2
    propcontrol.shockstep = 0.5
    propcontrol.endpoint = endpoint
    propcontrol.currentpos = 0.0
    propcontrol.storepos = [fel, faz]

    # material parameters
    propcontrol.material = material
    if isinstance(material, AbPhantom):
        propcontrol.d = 0.035
    else:
        propcontrol.d = 0.02
    propcontrol.offset = [0, 0]
    ns = 8
    propcontrol.numscreens = ns
    propcontrol.abamp = 0.09 * numpy.ones((ns, 1)) * 1e-3
    propcontrol.ablength = numpy.ones((ns, 1)) * 1e-3 * numpy.array([4, 100])
    propcontrol.abseed = numpy.arange(1, ns + 1)
    if config.heterogeneous_medium == AberrationFromFile:
        # TODO is it not a bool?
        propcontrol.abfile = 'randseq.mat'
    elif config.heterogeneous_medium == AberrationPhantom:
        # TODO is it not a bool?
        propcontrol.abfile = 'phantoml.mat'
    else:
        propcontrol.abfile = ''

    # signal parameters
    propcontrol.Fs = Fs
    propcontrol.dx = dx
    propcontrol.dy = dy
    propcontrol.dz = c / (2.0 * numpy.pi * fi)
    propcontrol.dt = dt
    propcontrol.fc = ft
    propcontrol.bandwidth = bandwidth * ft
    propcontrol.np = np
    propcontrol.amplitude = amp
    propcontrol.harmonic = harm
    propcontrol.filter = filter[0:harm]

    # transducer parameters
    propcontrol.Dx = daz
    propcontrol.Dy = _del
    propcontrol.Fx = faz
    propcontrol.Fy = fel
    if (new_config.diffraction_type == PseudoDifferential or
            new_config.diffraction_type == FiniteDifferenceTimeDifferenceReduced or
            new_config.diffraction_type == FiniteDifferenceTimeDifferenceFull) and new_config.annular_transducer:
        propcontrol.cchannel = numpy.array([1, 1])
    else:
        propcontrol.cchannel = numpy.floor([propcontrol.nx / 2, propcontrol.ny / 2]) + 1
    propcontrol.cchannel.astype(int)
    propcontrol.Nex = nelaz
    propcontrol.Ney = nelel
    propcontrol.esizex = dimelaz
    propcontrol.esizey = dimelel

    if new_config.heterogeneous_medium:
        propcontrol.storepos = numpy.array([propcontrol.storepos, propcontrol.d])
    propcontrol.storepos = numpy.unique(propcontrol.storepos)

    return propcontrol
