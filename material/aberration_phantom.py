"""
aberration_phantom.py
"""
from material.material import BaseMaterial


class AberrationPhantom(BaseMaterial):
    """
    AberrationPhantom
    """

    def __init__(self,
                 temperature: float):
        super().__init__(temperature)

    @property
    def constant_of_attenuation(self) -> float:
        """
        Get the constant of the attenuation
        :return: The constant of the attenuation
        """
        raise NotImplementedError

    @property
    def exponent_of_attenuation(self) -> float:
        """
        Get the exponent of the attenuation
        :return: The exponent of the attenuation
        """
        raise NotImplementedError

    @property
    def non_linearity_coefficient(self) -> float:
        """
        Get the exponent of the attenuation
        :return: The exponent of the attenuation
        """
        raise NotImplementedError

    @property
    def mass_density(self) -> float:
        """
        Get the density
        :return: The density
        """
        raise NotImplementedError

    @property
    def sound_speed(self) -> float:
        """
        Get the wave speed
        :return: wave speed
        """
        raise NotImplementedError
