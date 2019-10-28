from material.get_matparam import get_matparam
from consts import DENSITY

def get_density(material, temp):
    rho = get_matparam(material, DENSITY, temp)

    return rho
