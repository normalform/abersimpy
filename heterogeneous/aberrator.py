from initpropcontrol import initpropcontrol


def aberrator(propcontrol = None,
              phantom = None):
    if propcontrol is None:
        propcontrol = initpropcontrol()

    heterogeneous_medium = propcontrol.config.heterogeneous_medium
    if phantom is None:
        heterogeneous_medium = 1

    if heterogeneous_medium == 1 or heterogeneous_medium == 2:
        raise NotImplementedError
        #delta = aberrator_delayscreen(propcontrol)
    elif heterogeneous_medium == 3:
        raise NotImplementedError
        #delta = aberrator_phantom(propcontrol, phantom)
    else:
        delta = 0

    return delta