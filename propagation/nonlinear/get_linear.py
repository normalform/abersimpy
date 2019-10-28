import numpy


def get_linear(t,
               u,
               ti):
    nt2 = numpy.max(ti.shape)
    it2 = 2

    ui = []
    for ii in range(nt2):
        while ti[ii] > t[it2]:
            it2 = it2 + 1
        it1 = it2 - 1
        dpdt = (u[it2] - u[it1]) / (t[it2] - t[it1])
        ui.append(u[it1] + (ti[ii] - t[it1]) * dpdt)

    return ui
