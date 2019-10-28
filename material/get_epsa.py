from material.get_attconst import get_attconst
from material.get_attexp import get_attexp

import numpy


def get_epsa(material, temp):
    a = get_attconst(material, temp)
    b = get_attexp(material, temp)

    epsa = a / ((2.0 * numpy.pi) ** b * 20.0 * numpy.log10(numpy.exp(1)));

    return epsa
