from material.get_matparam import get_matparam
from consts import WAVESPEED

def get_wavespeed(material, temp):
    c = get_matparam(material, WAVESPEED, temp)

    return c
