from material.list_matrial import list_material


def check_material(mat):
    ret = []
    if isinstance(mat, int) or isinstance(mat, str):
        ret = list_material(mat)
        if not ret:
            print('Material must be specified in \'list_material\' for initiation by Abersim.')

    return ret
