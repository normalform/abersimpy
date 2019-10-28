from material.get_wavespeed import get_wavespeed
from material.get_density import get_density


def get_compressibility(material, temp):
    c = get_wavespeed(material, temp)
    rho = get_density(material, temp)
    kappa = 1. / (c**2.0 * rho)

    return kappa