from material.list_matrial import list_material
from material.get_epsn import get_epsn
from material.get_betan import get_betan
from material.get_epsa import get_epsa
from material.get_epsb import get_epsb
from material.get_wavespeed import get_wavespeed
from material.get_density import get_density
from material.get_compressibility import get_compressibility
from material.get_attconst import get_attconst
from material.get_attexp import get_attexp

from material.material import Material


def set_material(matname, temp = 37.0):
    mat = None
    if isinstance(matname, str):
        matidx = list_material(matname)
        mat = set_material(matidx, temp)
    else:
        assert isinstance(matname, int)
        mat = list_material(matname)
        if mat:
            mat = Material()
            mat.i = matname
            mat.temp = temp
            mat.eps[0] = get_epsn(matname, temp)
            mat.eps[1] = get_epsa(matname, temp)
            mat.eps[2] = get_epsb(matname, temp)
            mat.c0 = get_wavespeed(matname, temp)
            mat.rho = get_density(matname, temp)
            mat.kappa = get_compressibility(matname, temp)
            mat.betan = get_betan(matname, temp)
            mat.a = get_attconst(matname, temp)
            mat.b = get_attexp(matname, temp)

    return mat
