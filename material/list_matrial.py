WATER = 0
MUSCLE = 1

ABPHANTOM = 10

VDWGAS = 31

tissues = ['water', 'muscle', 'fat', 'liver', 'brain', 'bone', 'breast', 'blood']
phantoms = ['abphantom', 'std05', 'std03', 'surfplf', 'pipephantom', 'linmuscle']
liquids = ['seawater']
gases = ['air', 'vdwgas']
solids = ['rock', 'sandstone', 'sand']


def list_material(matname):
    mat = []

    if isinstance(matname, int):
        if matname >= 0 and matname <= 10:
            idx = matname
            if idx < len(tissues):
                mat = tissues[matname]
        elif matname > 10 and matname <= 20:
            idx = matname - 10
            if idx < len(phantoms):
                mat = phantoms[idx]
        elif matname > 20 and matname <= 30:
            idx = matname - 20
            if idx < len(liquids):
                mat = liquids[idx]
        elif matname > 30 and matname <= 40:
            idx = matname - 30
            if idx < len(gases):
                mat = gases[idx]
        elif matname > 40 and matname <= 50:
            idx = matname - 40
            if idx < len(solids):
                mat = solids[idx]

        if not mat:
            print('Material {} is not specified.'.format(matname))
            print('Please assign an identifier if this is an active material name.')
        else:
            mat = mat.capitalize()

    return mat
