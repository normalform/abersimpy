from material.get_matparam import get_matparam
from consts import BETAN

def get_betan(material, temp):
    betan = get_matparam(material, BETAN, temp)

    return betan