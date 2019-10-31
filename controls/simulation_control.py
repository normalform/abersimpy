"""
simulation_control.py
"""
import numpy

from controls.config_control import ConfigControl
from controls.domain_control import DomainControl
from controls.material_control import MaterialControl


class SimulationControl:
    """
    SimulationControl
    """

    def __init__(self,
                 domain: DomainControl,
                 end_point,
                 focus_elevation,
                 focus_azimuth,
                 config: ConfigControl,
                 material: MaterialControl):
        # simulation control parameters
        self.step_size = domain.step_size
        self.num_windows: int = 2
        self.shock_step: float = 0.5
        self.endpoint = end_point

        # TODO theses are variables. It should be in other place rather than this control.
        self.current_position: float = 0.0
        self.store_position = [focus_elevation, focus_azimuth]

        if config.heterogeneous_medium:
            self.store_position = numpy.array([self.store_position, material.thickness])
        self.store_position = numpy.unique(self.store_position)
