from material.get_matparam import get_matparam
from consts import MassDensityParamId

def get_density(material, temp):
    rho = get_matparam(material, MassDensityParamId, temp)

    return rho
