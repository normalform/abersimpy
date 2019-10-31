"""
material_control.py
"""
import numpy

from controls import consts
from controls.main_control import ConfigControl
from material.aberration_phantom import AberrationPhantom
from material.interfaces import IMaterial


class MaterialControl:
    """
    MaterialControl
    """

    def __init__(self,
                 material: IMaterial,
                 config: ConfigControl):
        # material control parameters
        self.material = material
        if isinstance(material, AberrationPhantom):
            self.thickness = 0.035
        else:
            self.thickness = 0.02
        self.offset = [0, 0]
        _num_screens = 8
        self.num_screens = _num_screens
        self.delay_screens_amplitude = 0.09 * numpy.ones((_num_screens, 1)) * 1e-3
        self.delay_screens_length = numpy.ones((_num_screens, 1)) * 1e-3 * numpy.array([4, 100])
        self.delay_screens_seed = numpy.arange(1, _num_screens + 1)
        if config.heterogeneous_medium == consts.AberrationFromFile:
            # TODO
            self.delay_screens_file = 'randseq.mat'
        elif config.heterogeneous_medium == consts.AberrationPhantom:
            # TODO
            self.delay_screens_file = 'phantoml.mat'
        else:
            self.delay_screens_file = ''
