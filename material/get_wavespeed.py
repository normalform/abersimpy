from material.get_matparam import get_matparam
from consts import WaveSpeedParamId

def get_wavespeed(material, temp):
    c = get_matparam(material, WaveSpeedParamId, temp)

    return c
