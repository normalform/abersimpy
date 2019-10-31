import numpy


def find_steps(z0,
               z1,
               step_size=None,
               store_position=numpy.array([]),
               screenpos=numpy.array([])):
    if step_size is None:
        print('Start, stop, and step_size must be specified')

    # set tolerance and initiate variables
    TOL = 1e-14
    specpos = numpy.unique(store_position, screenpos)
    z = z0
    step = numpy.zeros((int(numpy.ceil(((z1 - z0) / step_size) + numpy.max(specpos.shape))),))
    stepidx = numpy.zeros((int(numpy.ceil(((z1 - z0) / step_size) + numpy.max(specpos.shape))),), dtype=int)
    nsteps = 0

    # start loop
    k = 0
    q = numpy.where(specpos > z)[0][0]
    while z < z1:
        if q < numpy.max(specpos.shape):
            dspec = specpos[q] - z
        else:
            dspec = 100
        if dspec < step_size:
            if numpy.abs(dspec) < TOL:
                q = q + 1
                continue
            else:
                if numpy.abs(dspec - step_size) < TOL:
                    step[k] = step_size
                    stepidx[k] = 0
                else:
                    step[k] = dspec
                    stepidx[k] = -1
                    q = q + 1
                k = k + 1
        else:
            if dspec > step_size and numpy.abs(dspec - step_size) > TOL:
                stepn = numpy.ceil(numpy.sum(step / step_size))
                dspec = stepn * step_size - z
                stepidx[k] = 0
                if dspec < TOL:
                    dspec = step_size
                    stepidx[k] = 0
                step[k] = dspec
            else:
                step[k] = step_size
                stepidx[k] = 0
            k = k + 1
        if z + step[k] > z1:
            if numpy.abs(z + step[k] - z1) > TOL:
                if numpy.abs(z1 - z) < TOL:
                    step = step[:k]
                    stepidx = step[:k]
                    nsteps = k - 2
                    break
                step[k] = z1 - z
                stepidx[k] = 0
                nsteps = k
                break
        z = z0 + numpy.sum(step / step_size) * step_size
        nsteps = k

    # adjust size of vectors
    step = step[:nsteps]
    stepidx = stepidx[:nsteps]

    return nsteps, step, stepidx
