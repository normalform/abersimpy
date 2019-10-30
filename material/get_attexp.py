from material.get_matparam import get_matparam
from consts import ExponentOfAttenuationParamId

def get_attexp(material, temp):
    b = get_matparam(material, ExponentOfAttenuationParamId, temp)

    return b