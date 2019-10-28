from material.get_matparam import get_matparam
from consts import ATTEXP

def get_attexp(material, temp):
    b = get_matparam(material, ATTEXP, temp)

    return b