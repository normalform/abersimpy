"""
body_wall.py
"""
import numpy

from controls.consts import AberrationFromDelayScreenBodyWall
from controls.main_control import MainControl
from heterogeneous.aberration import aberration
from propagation.get_wave_numbers import get_wave_numbers


def body_wall(control: MainControl,
              signal: numpy.ndarray,
              direction: int,
              equidistant_steps: bool,
              wave_numbers: numpy.ndarray = None,
              window: numpy.ndarray = None,
              phantom=None):
    """
    Propagation through a body wall consisting of equidistant delay screens or phantom.
    TODO: Need more implementation
    TODO: Need unit tests
    :param control: Controls.
    :param signal: The signal to be propagated.
    :param direction: Sign determines in positive or negative z-dir.
    :param equidistant_steps: The flag specifying beam simulation with equidistant steps.
    :param wave_numbers: Wave number operator.
    :param window: Window for tapering the solution to zero close to the boundaries.
        If not specified a window of with a zero region of length 2 * step_size and a raised cosine
        tapering of length 1.5 * step_size will be used. If window is set to -1, no windowing
        will be performed.
    :param phantom: Phantom is a function handle to a function who's input is a (x,y,z)
        coordinate and output is the local wave speed at the coordinate.
        The phantom is initialized by a function call with one string argument and finalized with a
        function call with no input arguments. Initialization is performed by the function.
    :return: Wave field at the end of the simulation,
             Temporal RMS beam profile for all frequencies.
                The profile has dimensions (num_points_y * num_points_x * num_steps * harmonic),
                possibly with num_points_y as singleton dimension. The index of the
                harmonic-dimension gives the n'th harmonic profile starting with 0 as the
                total field.
             Maximum pressure in the temporal direction. The profile is structured as the
                RMS profile. The maximum is found as the maximum of the envelope of the signal
                at each point in space.
             Signal located at the control.transducer.center_channel index in space.
                This is the raw signal without any filtering performed for each step.
             The z-coordinate of each profile and axial pulse.
    """
    if wave_numbers is not None:
        _wave_numbers = get_wave_numbers(control, equidistant_steps)
    else:
        _wave_numbers = wave_numbers

    # Initiates profiles
    rmspro = None
    maxpro = None
    axplse = None
    zpos = None

    # Initiate variables
    num_dimensions = control.num_dimensions
    num_points_x = control.domain.num_points_x
    num_points_y = control.domain.num_points_y
    num_points_t = control.domain.num_points_t
    heterogeneous_medium = control.heterogeneous_medium
    annular_transducer = control.annular_transducer
    resolution_x = control.signal.resolution_x
    resolution_y = control.signal.resolution_y

    step_size = control.simulation.step_size
    ns = control.material.num_screens
    thickness = control.material.thickness
    endpoint = numpy.min(control.simulation.endpoint, thickness)

    # adjust step sizes
    if heterogeneous_medium == AberrationFromDelayScreenBodyWall or phantom is None:
        if phantom is None:
            print('Changing heterogeneous_medium to AberrationFromDelayScreenBodyWall')
            heterogeneous_medium = AberrationFromDelayScreenBodyWall
            control.heterogeneous_medium = heterogeneous_medium

    dscreen = thickness / ns
    nsubsteps = numpy.ceil(dscreen / step_size)
    abstepsize = dscreen / nsubsteps
    control.simulation.step_size = abstepsize

    # Prepares body wall model
    delta = aberration(control)

    raise NotImplementedError
