"""
interfaces.py

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
from abc import ABC, abstractmethod


class IMaterial(ABC):
    """
    IMaterial
    """

    @property
    @abstractmethod
    def constant_of_attenuation(self) -> float:
        """
        Get the constant of the attenuation
        :return: The constant of the attenuation
        """

    @property
    @abstractmethod
    def exponent_of_attenuation(self) -> float:
        """
        Get the exponent of the attenuation
        :return: The exponent of the attenuation
        """

    @property
    @abstractmethod
    def non_linearity_coefficient(self) -> float:
        """
        Get the exponent of the attenuation
        :return: The exponent of the attenuation
        """

    @property
    @abstractmethod
    def mass_density(self) -> float:
        """
        Get the density
        :return: The density
        """

    @property
    @abstractmethod
    def sound_speed(self) -> float:
        """
        Get the sound speed
        :return: sound speed
        """

    @property
    @abstractmethod
    def eps_a(self) -> float:
        """
        Get the coefficient eps_a of material
        :return: The coefficient eps_a of material
        """

    @property
    @abstractmethod
    def eps_b(self) -> float:
        """
        Get the coefficient eps_b of material
        :return: The coefficient eps_b of material
        """

    @property
    @abstractmethod
    def eps_n(self) -> float:
        """
        Get the coefficient eps_n of material
        :return: The coefficient eps_n of material
        """

    @property
    @abstractmethod
    def compressibility(self) -> float:
        """
        Get the compressibility
        :return: The compressibility
        """

    @property
    @abstractmethod
    def is_regular(self) -> bool:
        """
        Returns True if it's regular material otherwise returns False
        :return:
        """
