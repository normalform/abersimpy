from material.material import Material
from material.list_matrial import VDWGAS
from consts import ScaleForSpatialVariablesZ, ScaleForPressure, ScaleForTemporalVariable
from material.get_betan import get_betan
from material.get_compressibility import get_compressibility
from material.get_wavespeed import get_wavespeed

import numpy


def get_epsn(material, temp = 37.0):
    epsn = 0.0

    if isinstance(material, Material):
        betan = material.betan
        kappa = material.kappa
        c = material.c0
    else:
        betan = get_betan(material, temp)
        kappa = get_compressibility(material, temp)
        c = get_wavespeed(material, temp)
        if material == VDWGAS:
            epsn = 1.0
            return epsn

    epsn = numpy.divide(numpy.multiply(betan, kappa), c)
    epsn = (ScaleForSpatialVariablesZ * ScaleForPressure / ScaleForTemporalVariable) * epsn

    return epsn