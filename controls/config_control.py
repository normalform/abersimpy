"""
config_control.py
"""
from diffraction.interfaces import IDiffractionType


class ConfigControl:
    """
    ConfigControl
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
        msg = 'ConfigControl('
        msg += f'diffraction type:{self.diffraction_type}, '
        msg += f'non-linearity:{self.non_linearity}, '
        msg += f'attenuation:{self.attenuation}, '
        msg += f'heterogeneous_medium:{self.heterogeneous_medium}, '
        msg += f'annular transducer:{self.annular_transducer}, '
        msg += f'equidistant steps:{self.equidistant_steps}, '
        msg += f'history:{self.history})'

        return msg
