from material.get_matparam import get_matparam
from consts import ConstantOfAttenuationParamId

def get_attconst(material, temp):
    a = get_matparam(material, ConstantOfAttenuationParamId, temp)

    return a
