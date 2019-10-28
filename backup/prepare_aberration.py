from backup.prepare_delayscreen import prepare_delayscreen


def prepare_aberration(delta, phantom, prop, logstr):
    abflag = 0
    data = None

    abflag = prop.abflag

    if abflag == 0:
        pass
    elif abflag == 1:
        prepare_delayscreen(delta, data, prop, logstr)
    elif abflag == 2:
        prepare_delayscreen(delta, data, prop, logstr)
    elif abflag == 3:
        print('Aberration type not publicly available')

    print('Aberration and body wall prepared')
