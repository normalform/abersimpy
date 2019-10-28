from material.get_matparam import get_matparam
from consts import ATTCONST

def get_attconst(material, temp):
    a = get_matparam(material, ATTCONST, temp)

    return a
