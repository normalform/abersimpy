from material.get_matparam import get_matparam
from consts import NonLinearityCoefficientParamId

def get_betan(material, temp):
    betan = get_matparam(material, NonLinearityCoefficientParamId, temp)

    return betan