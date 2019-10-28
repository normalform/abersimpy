from initpropcontrol import initpropcontrol


def aberrator(propcontrol = None,
              phantom = None):
    if propcontrol is None:
        propcontrol = initpropcontrol()

    abflag = propcontrol.abflag
    if phantom is None:
        abflag = 1

    if abflag == 1 or abflag == 2:
        raise NotImplementedError
        #delta = aberrator_delayscreen(propcontrol)
    elif abflag == 3:
        raise NotImplementedError
        #delta = aberrator_phantom(propcontrol, phantom)
    else:
        delta = 0

    return delta