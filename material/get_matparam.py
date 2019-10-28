from material.list_matrial import list_material
from material.muscle import muscle


def get_matparam(mat, param, temp):
    mat = list_material(mat)
    value = 0.0
    if mat == 'Muscle':
        value = muscle(param, temp)

    return value